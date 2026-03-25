from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from plant_3d.application.types import LabeledPointCloud


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_stl_points(path: str | Path) -> np.ndarray:
    mesh = trimesh.load_mesh(str(_as_path(path)))
    return np.asarray(mesh.vertices, dtype=np.float64)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, keep_idx = np.unique(keys, axis=0, return_index=True)
    keep_idx.sort()
    return points[keep_idx]


def save_pcd_ascii(path: str | Path, points: np.ndarray) -> Path:
    path_obj = _as_path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
        f.write(f"WIDTH {points.shape[0]}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {points.shape[0]}\nDATA ascii\n")
        for x, y, z in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
    return path_obj


def load_pcd_ascii(path: str | Path) -> np.ndarray:
    path_obj = _as_path(path)
    data_lines: list[str] = []
    in_data = False
    with path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            if in_data:
                data_lines.append(raw)
                continue
            if raw.upper().startswith("DATA"):
                in_data = True
    if not data_lines:
        return np.zeros((0, 3), dtype=np.float64)
    pts: list[list[float]] = []
    for row in data_lines:
        tokens = row.split()
        if len(tokens) < 3:
            continue
        pts.append([float(tokens[0]), float(tokens[1]), float(tokens[2])])
    return np.asarray(pts, dtype=np.float64)


def _default_palette() -> dict[int, tuple[int, int, int]]:
    return {0: (128, 128, 128), 1: (40, 180, 99), 2: (52, 152, 219), 3: (231, 76, 60)}


def labels_to_colors(labels: np.ndarray, class_map: dict[int, tuple[int, int, int]] | None = None) -> np.ndarray:
    cmap = class_map or _default_palette()
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for i, label in enumerate(labels.astype(int)):
        colors[i] = cmap.get(label, (255, 255, 255))
    return colors


def parse_labels_txt(path: str | Path) -> np.ndarray:
    lines = _as_path(path).read_text(encoding="utf-8").splitlines()
    vals = [int(x.strip()) for x in lines if x.strip()]
    return np.asarray(vals, dtype=np.int32)


def save_labeled_ply(path: str | Path, cloud: LabeledPointCloud) -> Path:
    out = _as_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    colors = cloud.colors if cloud.colors is not None else labels_to_colors(cloud.labels)
    pc = trimesh.PointCloud(cloud.points, colors=colors)
    pc.export(str(out))
    return out


def load_labeled_ply(path: str | Path, labels_txt: str | Path | None = None) -> LabeledPointCloud:
    p = _as_path(path)
    if p.suffix.lower() == ".pcd":
        points = load_pcd_ascii(p)
        if labels_txt is None:
            raise ValueError("labels_txt required for .pcd input")
        labels = parse_labels_txt(labels_txt)
        return LabeledPointCloud(points=points, labels=labels)
    loaded = trimesh.load(str(p))
    if isinstance(loaded, trimesh.Scene):
        geom = loaded.dump(concatenate=True)
    else:
        geom = loaded
    points = np.asarray(geom.vertices, dtype=np.float64)
    if labels_txt is not None:
        labels = parse_labels_txt(labels_txt)
    elif hasattr(geom, "visual") and hasattr(geom.visual, "vertex_colors"):
        colors = np.asarray(geom.visual.vertex_colors[:, :3], dtype=np.uint8)
        labels = np.zeros(points.shape[0], dtype=np.int32)
        return LabeledPointCloud(points=points, labels=labels, colors=colors)
    else:
        labels = np.zeros(points.shape[0], dtype=np.int32)
    return LabeledPointCloud(points=points, labels=labels)


def save_label_metadata(path: str | Path, labels: np.ndarray, class_map: dict[int, tuple[int, int, int]]) -> Path:
    out = _as_path(path)
    payload = {
        "count": int(labels.shape[0]),
        "classes": {str(k): {"rgb": list(v)} for k, v in class_map.items()},
        "label_histogram": {str(k): int((labels == k).sum()) for k in np.unique(labels)},
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
