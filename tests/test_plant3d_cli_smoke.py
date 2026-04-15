from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from click.testing import CliRunner

from plant_3d.cli import main
from plant_3d.application.types import LabeledPointCloud
from plant_3d.infrastructure.io.formats import save_labeled_ply


def test_cli_help() -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["--help"])
    assert res.exit_code == 0
    assert "plant3d" in res.output


def test_cli_morph_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    pts = np.array([[0, 0, 0], [0, 0, 0.1], [0.1, 0, 0.1], [0.2, 0, 0.1]], dtype=np.float64)
    labels = np.array([1, 1, 2, 2], dtype=np.int32)
    ply = save_labeled_ply(tmp_path / "seg.ply", LabeledPointCloud(points=pts, labels=labels))
    res = runner.invoke(main, ["morph", str(ply), "-o", str(tmp_path / "out")])
    assert res.exit_code == 0, res.output
