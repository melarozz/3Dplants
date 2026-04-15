from __future__ import annotations

from pathlib import Path

import numpy as np

from plant_3d.application.types import LabeledPointCloud
from plant_3d.infrastructure.morphometrics.metrics import compute_morphometrics


def test_compute_morphometrics_generates_reports(tmp_path: Path) -> None:
    stem = np.array([[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.3]], dtype=np.float64)
    leaf = np.array([[0, 0, 0.15], [0.08, 0.01, 0.16], [0.16, 0.01, 0.17], [0.12, 0.02, 0.18], [0.04, -0.01, 0.14]], dtype=np.float64)
    points = np.vstack([stem, leaf])
    labels = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.int32)
    cloud = LabeledPointCloud(points=points, labels=labels)
    out = compute_morphometrics(cloud, tmp_path, stem_label=1)
    assert out.csv_path.exists()
    assert "leaf_label" in out.json_path.read_text(encoding="utf-8")


def test_compute_morphometrics_supports_displacement_and_speed(tmp_path: Path) -> None:
    points = np.array([[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.15], [0.1, 0, 0.16], [0.2, 0, 0.17]], dtype=np.float64)
    labels = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
    cloud = LabeledPointCloud(points=points, labels=labels)
    vectors = np.zeros_like(points)
    vectors[labels == 2] = np.array([0.05, 0, 0], dtype=np.float64)
    vec_path = tmp_path / "vectors.npz"
    np.savez(vec_path, points=points, labels=labels, vectors=vectors)
    out = compute_morphometrics(cloud, tmp_path / "m", stem_label=1, displacement_npz=vec_path, delta_time_hours=10, skeleton_bins=5)
    assert "growth_speed_per_hour" in out.json_path.read_text(encoding="utf-8")
