from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from plant_3d.infrastructure.metric_calibration import (
    calibrate_files,
    compute_scale_from_reference_extent,
    extent_along_axis,
    unit_to_meters,
)


def test_unit_to_meters() -> None:
    assert unit_to_meters(100, "cm") == 1.0
    assert unit_to_meters(1, "m") == 1.0


def test_scale_from_reference(tmp_path: Path) -> None:
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    k, ext = compute_scale_from_reference_extent(v, target_extent_m=0.15, axis="z")
    assert ext == 1.0
    assert np.isclose(k, 0.15)


def test_calibrate_reference_mode(tmp_path: Path) -> None:
    ref = tmp_path / "ref.stl"
    b = trimesh.creation.box(extents=(0.2, 0.2, 1.0))
    b.export(str(ref))
    other = tmp_path / "o.stl"
    b.export(str(other))
    out_dir = tmp_path / "out"
    written, summary = calibrate_files(
        [other], out_dir, reference_path=ref, physical_extent=12.0, unit="cm", axis="z",
        uniform_scale=None, shift_min_to_zero=False, name_suffix="m",
    )
    assert len(written) == 1
    assert summary.mode == "reference_extent"
    mesh = trimesh.load(str(written[0]))
    assert np.isclose(extent_along_axis(np.asarray(mesh.vertices), 2), 0.12, rtol=1e-3)


def test_calibrate_uniform(tmp_path: Path) -> None:
    p = tmp_path / "a.stl"
    trimesh.creation.box(extents=(1, 1, 1)).export(str(p))
    written, summary = calibrate_files(
        [p], tmp_path / "out", reference_path=None, physical_extent=None, unit="cm", axis="z",
        uniform_scale=2.0, shift_min_to_zero=False, name_suffix="x",
    )
    assert summary.mode == "uniform_scale"
    assert summary.scale_factor == 2.0
