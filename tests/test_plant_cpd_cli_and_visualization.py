from __future__ import annotations

from pathlib import Path

import numpy as np
from click.testing import CliRunner

import plant_3d.infrastructure.cpd.cli as cpd_cli


def test_plant_cpd_cli_help() -> None:
    runner = CliRunner()
    res = runner.invoke(cpd_cli.main, ["--help"])
    assert res.exit_code == 0
    assert "plant-cpd" in res.output


def test_plant_cpd_cli_register_smoke(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "s.glb"
    tgt = tmp_path / "t.glb"
    src.write_bytes(b"glb")
    tgt.write_bytes(b"glb")

    class _R:
        def __init__(self, out_dir: str):
            self.output_dir = out_dir
            self.metrics = {"rmse": 0.1}

    import plant_3d.infrastructure.cpd.pipeline as pipeline
    monkeypatch.setattr(pipeline, "register", lambda **k: _R(str(tmp_path / "out")))

    runner = CliRunner()
    res = runner.invoke(cpd_cli.main, ["register", str(src), str(tgt), "-o", str(tmp_path / "out"), "--rigid-only", "--no-hdf5"])
    assert res.exit_code == 0, res.output
