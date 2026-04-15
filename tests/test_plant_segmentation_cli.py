from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from click.testing import CliRunner

from plant_3d.infrastructure.segmentation.cli.segment import main as seg_main


def test_plant_segment_cli(tmp_path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1)
    p = tmp_path / "m.stl"
    mesh.export(str(p))
    out = tmp_path / "labels.npy"
    runner = CliRunner()
    res = runner.invoke(seg_main, [str(p), "--output", str(out)])
    assert res.exit_code == 0
    assert out.exists()
