from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import trimesh

from plant_3d.application.models import MorphResult
from plant_3d.application.types import LabeledPointCloud


def _principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    n = np.linalg.norm(axis)
    if n > 0:
        return axis / n
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _compute_skeleton(points: np.ndarray, *, n_bins: int = 30) -> np.ndarray:
    if points.shape[0] < 3:
        return points.copy()
    axis = _principal_axis(points)
    proj = points @ axis
    n_points = int(points.shape[0])
    n_bins = int(np.clip(n_bins, 3, max(3, n_points // 2)))
    edges = np.linspace(float(proj.min()), float(proj.max()), n_bins + 1)
    skeleton_pts: list[np.ndarray] = []
    for b in range(n_bins):
        if b == n_bins - 1:
            in_bin = (proj >= edges[b]) & (proj <= edges[b + 1])
        else:
            in_bin = (proj >= edges[b]) & (proj < edges[b + 1])
        if not np.any(in_bin):
            continue
        skeleton_pts.append(points[in_bin].mean(axis=0))
    if len(skeleton_pts) < 2:
        return points.copy()
    return np.asarray(skeleton_pts, dtype=np.float64)


def _skeleton_length(skeleton: np.ndarray) -> float:
    if skeleton.shape[0] < 2:
        return 0.0
    diffs = np.diff(skeleton, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _leaf_area(points: np.ndarray) -> float:
    if points.shape[0] < 4:
        return 0.0
    hull = trimesh.points.PointCloud(points).convex_hull
    return float(np.asarray(hull.area_faces, dtype=np.float64).sum())


def _angle_between_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def compute_morphometrics(
    cloud: LabeledPointCloud,
    output_dir: str | Path,
    stem_label: int = 1,
    *,
    displacement_npz: str | Path | None = None,
    delta_time_hours: float | None = None,
    skeleton_bins: int = 30,
) -> MorphResult:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem_points = cloud.points[cloud.labels == stem_label]
    stem_axis = _principal_axis(stem_points) if stem_points.shape[0] >= 3 else np.array([0.0, 0.0, 1.0])

    disp_vectors = None
    if displacement_npz:
        payload = np.load(Path(displacement_npz).expanduser().resolve())
        disp_vectors = np.asarray(payload["vectors"], dtype=np.float64)
        if disp_vectors.shape != cloud.points.shape:
            raise ValueError(
                f"Displacement vectors shape mismatch: {disp_vectors.shape} vs {cloud.points.shape}",
            )
        if delta_time_hours is not None and delta_time_hours <= 0:
            raise ValueError("delta_time_hours must be > 0 when provided")

    labels_sorted = sorted(set(cloud.labels.astype(int).tolist()))
    leaf_axes: dict[int, np.ndarray] = {}
    for label in labels_sorted:
        if label in (0, stem_label):
            continue
        pts = cloud.points[cloud.labels == label]
        if pts.shape[0] >= 3:
            leaf_axes[label] = _principal_axis(pts)

    rows: list[dict[str, object]] = []
    for label in labels_sorted:
        if label in (0, stem_label):
            continue
        pts = cloud.points[cloud.labels == label]
        if pts.shape[0] < 3:
            continue
        skeleton = _compute_skeleton(pts, n_bins=skeleton_bins)
        length = _skeleton_length(skeleton)
        area = _leaf_area(pts)
        leaf_axis = leaf_axes.get(label, _principal_axis(pts))
        angle_to_stem = _angle_between_deg(leaf_axis, stem_axis)
        row: dict[str, object] = {
            "leaf_label": int(label),
            "surface_area": float(area),
            "leaf_length": float(length),
            "inclination_angle_deg": float(angle_to_stem),
            "points_count": int(pts.shape[0]),
        }
        if disp_vectors is not None:
            idx = np.where(cloud.labels == label)[0]
            mags = np.linalg.norm(disp_vectors[idx], axis=1) if idx.size else np.zeros((0,), dtype=np.float64)
            mean_disp = float(mags.mean()) if mags.size else 0.0
            row["mean_displacement"] = mean_disp
            row["max_displacement"] = float(mags.max()) if mags.size else 0.0
            if delta_time_hours is not None:
                row["growth_speed_per_hour"] = float(mean_disp / float(delta_time_hours))
        rows.append(row)

    csv_path = output / "morph_metrics.csv"
    fieldnames = [
        "leaf_label",
        "surface_area",
        "leaf_length",
        "inclination_angle_deg",
        "points_count",
        "mean_displacement",
        "max_displacement",
        "growth_speed_per_hour",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    pairwise: dict[str, float] = {}
    leaf_ids = sorted(leaf_axes.keys())
    for i, a in enumerate(leaf_ids):
        for b in leaf_ids[i + 1 :]:
            pairwise[f"{a}-{b}"] = float(_angle_between_deg(leaf_axes[a], leaf_axes[b]))

    json_path = output / "morph_metrics.json"
    json_path.write_text(
        json.dumps({"leaves": rows, "pairwise_leaf_angles_deg": pairwise}, indent=2),
        encoding="utf-8",
    )
    return MorphResult(csv_path=csv_path, json_path=json_path)
