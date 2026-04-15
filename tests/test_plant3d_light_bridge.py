from __future__ import annotations

from pathlib import Path

import trimesh

from plant_3d.infrastructure.light_bridge import run_light_simulation


def test_light_bridge_outputs(tmp_path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.05)
    glb_path = tmp_path / "plant.glb"
    mesh.export(glb_path)
    result = run_light_simulation(input_glb=glb_path, output_dir=tmp_path / "light", power=10)
    assert result.output_glb.exists()
    assert result.output_hdf5.exists()
    assert result.mean_ppfd >= 0
