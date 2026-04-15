from __future__ import annotations

from pathlib import Path

import trimesh
from click.testing import CliRunner

from plant_3d.cli import main


def test_cli_prep_seg_smoke(tmp_path: Path) -> None:
    stl = tmp_path / "m.stl"
    trimesh.creation.icosphere(subdivisions=1).export(str(stl))
    runner = CliRunner()
    res = runner.invoke(main, ["prep-seg", str(stl), "-o", str(tmp_path / "out")])
    assert res.exit_code == 0, res.output


def test_cli_light_smoke(tmp_path: Path) -> None:
    glb = tmp_path / "m.glb"
    trimesh.creation.icosphere(subdivisions=1).export(str(glb))
    runner = CliRunner()
    res = runner.invoke(main, ["light", str(glb), "-o", str(tmp_path / "light")])
    assert res.exit_code == 0, res.output
