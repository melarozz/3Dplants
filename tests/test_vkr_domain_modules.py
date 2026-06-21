from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import trimesh
from click.testing import CliRunner

from plant_3d.application.manifest import build_manifest, sha256_file
from plant_3d.application.photo_calibration import estimate_plant_height_from_photo, mesh_scale_from_photo
from plant_3d.application.preprocess import center_rgba_canvas
from plant_3d.infrastructure.metric_calibration import calibrate_from_photo_measurements
from plant_3d.interfaces.cli import main


def test_center_rgba_canvas_crops_and_centers() -> None:
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    rgba[2:5, 3:7, 3] = 255
    out = center_rgba_canvas(rgba, target_size=8)
    assert out.shape == (8, 8, 4)


def test_photo_calibration_formulas_match_vkr() -> None:
    est = estimate_plant_height_from_photo(pot_height_px=188.0, plant_height_px=263.0, pot_height_phys=12.5, unit="cm")
    assert est.plant_height_phys == pytest.approx(12.5 / 188.0 * 263.0)
    assert mesh_scale_from_photo(plant_height_phys=est.plant_height_phys, mesh_extent_norm=1.0) == pytest.approx(est.plant_height_phys)


def test_manifest_includes_sha256(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    manifest = build_manifest(command="prep-seg", profile_name="research", profile=None, profile_config=None, resolved_params={"input": str(p)}, artifacts={}, input_paths=[p])
    assert manifest["inputs"][0]["sha256"] == sha256_file(p)


def test_calibrate_from_photo_measurements(tmp_path: Path) -> None:
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 1.0))
    src = tmp_path / "plant.stl"
    mesh.export(str(src))
    written, summary, photo = calibrate_from_photo_measurements([src], tmp_path / "out", pot_height_px=188.0, plant_height_px=263.0)
    assert written and written[0].exists()
    assert summary.mode == "photo_extent"


def test_generate_3d_stub_command(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PLANT3D_HUNYUAN_STUB", "1")
    img = tmp_path / "front.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    runner = CliRunner()
    result = runner.invoke(main, ["generate-3d", str(img), "-o", str(tmp_path / "gen")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "gen" / "model.stl").exists()
