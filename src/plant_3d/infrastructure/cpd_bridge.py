from __future__ import annotations

import tempfile
from pathlib import Path

import trimesh

from plant_3d.infrastructure.cpd.config import PipelineConfig
from plant_3d.infrastructure.cpd.pipeline import RegistrationResult, register


def _to_glb_pointcloud(input_path: str | Path, tmp_path: str | Path) -> Path:
    geom = trimesh.load(str(Path(input_path).expanduser().resolve()))
    points = geom.vertices
    cloud = trimesh.PointCloud(points)
    scene = trimesh.Scene([cloud])
    out = Path(tmp_path).expanduser().resolve()
    scene.export(str(out), file_type="glb")
    return out


def register_any_format(
    source_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig | None = None,
) -> RegistrationResult:
    source_suffix = Path(source_path).suffix.lower()
    target_suffix = Path(target_path).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="plant3d_cpd_") as tmp_dir:
        tmp = Path(tmp_dir)
        src = Path(source_path).expanduser().resolve()
        tgt = Path(target_path).expanduser().resolve()
        if source_suffix != ".glb":
            src = _to_glb_pointcloud(source_path, tmp / "source.glb")
        if target_suffix != ".glb":
            tgt = _to_glb_pointcloud(target_path, tmp / "target.glb")
        return register(str(src), str(tgt), config=config, output_dir=output_dir)
