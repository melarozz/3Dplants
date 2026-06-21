from __future__ import annotations

from pathlib import Path

import trimesh

from plant_3d.infrastructure.io.formats import load_pcd_ascii
from plant_3d.infrastructure.seg_prep.session import run_segmentation_session


def test_segmentation_session_without_editor_runs_post(tmp_path: Path) -> None:
    stl_path = tmp_path / "input.stl"
    trimesh.creation.icosphere(subdivisions=1, radius=0.1).export(str(stl_path))
    pts = load_pcd_ascii(run_segmentation_session(input_stl=stl_path, output_dir=tmp_path / "sess", voxel_size=0.005, class_map={0: (128, 128, 128), 1: (0, 255, 0)}, run_editor=False).prep.pcd_path)
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("\n".join(["1"] * int(pts.shape[0])), encoding="utf-8")
    result = run_segmentation_session(input_stl=stl_path, output_dir=tmp_path / "sess2", voxel_size=0.005, class_map={0: (128, 128, 128), 1: (0, 255, 0)}, labels_txt=labels_path, run_editor=False)
    assert result.post is not None
    assert result.post.ply_path.exists()
