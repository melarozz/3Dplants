from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from plant_3d.application.models import TrackResult
from plant_3d.application.types import LabeledPointCloud
from plant_3d.infrastructure.io.formats import save_labeled_ply


def _centroid(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0)


def _match_labels(source: LabeledPointCloud, target: LabeledPointCloud) -> list[tuple[int, int]]:
    source_ids = sorted(set(source.labels.astype(int).tolist()) - {0})
    target_ids = sorted(set(target.labels.astype(int).tolist()) - {0})
    if not source_ids or not target_ids:
        return []
    target_centroids = {tid: _centroid(target.points[target.labels == tid]) for tid in target_ids}
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for sid in source_ids:
        spts = source.points[source.labels == sid]
        sc = _centroid(spts)
        best = None
        best_d = float("inf")
        for tid, tc in target_centroids.items():
            if tid in used:
                continue
            d = float(np.linalg.norm(sc - tc))
            if d < best_d:
                best_d = d
                best = tid
        if best is not None:
            used.add(best)
            pairs.append((sid, best))
    return pairs


def _compute_vectors(source: LabeledPointCloud, target: LabeledPointCloud, pairs: list[tuple[int, int]]) -> np.ndarray:
    vectors = np.zeros_like(source.points)
    for sid, tid in pairs:
        s_idx = np.where(source.labels == sid)[0]
        t_pts = target.points[target.labels == tid]
        if s_idx.size == 0 or t_pts.shape[0] == 0:
            continue
        tree = cKDTree(t_pts)
        _, nn_idx = tree.query(source.points[s_idx], k=1)
        mapped = t_pts[nn_idx]
        vectors[s_idx] = mapped - source.points[s_idx]
    return vectors


def _magnitude_to_colors(mag: np.ndarray) -> np.ndarray:
    if mag.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    mmax = float(mag.max()) if float(mag.max()) > 0 else 1.0
    norm = (mag / mmax).clip(0, 1)
    red = (255 * norm).astype(np.uint8)
    blue = (255 * (1 - norm)).astype(np.uint8)
    green = np.zeros_like(red)
    return np.stack([red, green, blue], axis=1)


def track_displacement(
    source: LabeledPointCloud,
    target: LabeledPointCloud,
    output_dir: str | Path,
    mode: str,
) -> TrackResult:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs = _match_labels(source, target)
    vectors = _compute_vectors(source, target, pairs)
    magnitudes = np.linalg.norm(vectors, axis=1)
    heat_colors = _magnitude_to_colors(magnitudes)
    vectors_path = output / "vectors.npz"
    np.savez(vectors_path, points=source.points, labels=source.labels, vectors=vectors)
    heatmap_path = output / "growth_heatmap.ply"
    save_labeled_ply(
        heatmap_path,
        LabeledPointCloud(points=source.points, labels=source.labels, colors=heat_colors),
    )
    report = {
        "mode": mode,
        "matched_organs": [{"source": s, "target": t} for s, t in pairs],
        "mean_displacement": float(magnitudes.mean()) if magnitudes.size else 0.0,
        "max_displacement": float(magnitudes.max()) if magnitudes.size else 0.0,
        "points_count": int(source.points.shape[0]),
    }
    report_path = output / "tracking_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return TrackResult(vectors_path=vectors_path, heatmap_path=heatmap_path, report_path=report_path)
