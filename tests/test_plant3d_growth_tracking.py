from __future__ import annotations

from pathlib import Path

import numpy as np

from plant_3d.application.types import LabeledPointCloud
from plant_3d.infrastructure.growth_tracking.tracking import track_displacement


def test_track_displacement_writes_outputs(tmp_path: Path) -> None:
    src = LabeledPointCloud(points=np.array([[0, 0, 0], [0.1, 0, 0]], dtype=np.float64), labels=np.array([1, 2], dtype=np.int32))
    tgt = LabeledPointCloud(points=np.array([[0.01, 0, 0], [0.11, 0, 0]], dtype=np.float64), labels=np.array([1, 2], dtype=np.int32))
    result = track_displacement(src, tgt, tmp_path, mode="track-growth")
    assert result.vectors_path.exists()
    assert result.heatmap_path.exists()
