from __future__ import annotations

from pathlib import Path

import trimesh

from plant_3d.infrastructure.seg_prep.pipeline import run_prep_seg


def test_run_prep_seg(tmp_path: Path) -> None:
    stl = tmp_path / "m.stl"
    trimesh.creation.icosphere(subdivisions=1).export(str(stl))
    result = run_prep_seg(input_stl=stl, output_dir=tmp_path / "out", voxel_size=0.01)
    assert result.pcd_path.exists()
    assert result.points_count > 0
