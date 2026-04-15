from __future__ import annotations

from pathlib import Path

import trimesh

from plant_3d.infrastructure.seg_prep.session import run_segmentation_session


def test_session_prep_only(tmp_path: Path) -> None:
    stl = tmp_path / "m.stl"
    trimesh.creation.icosphere(subdivisions=1).export(str(stl))
    result = run_segmentation_session(input_stl=stl, output_dir=tmp_path / "s", voxel_size=0.01, class_map={1: (0, 255, 0)}, run_editor=False)
    assert result.prep.pcd_path.exists()
    assert result.post is None
