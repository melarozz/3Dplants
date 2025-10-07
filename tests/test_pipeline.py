"""Integration tests for the registration pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import trimesh

from plant_cpd.config import PipelineConfig, PreprocessConfig
from plant_cpd.pipeline import register


def _create_test_glb(
    path: Path,
    points: np.ndarray,
) -> Path:
    """Create a minimal GLB file from points."""
    # Create a mesh by using convex hull
    cloud = trimesh.PointCloud(points)
    mesh = cloud.convex_hull
    scene = trimesh.Scene([mesh])
    scene.export(str(path), file_type="glb")
    return path


class TestPipeline:
    def test_full_pipeline(
        self, sphere_cloud, transformed_sphere, tmp_path,
    ):
        """Full pipeline: load → preprocess → rigid → nonrigid."""
        src_path = _create_test_glb(
            tmp_path / "source.glb", transformed_sphere,
        )
        tgt_path = _create_test_glb(
            tmp_path / "target.glb", sphere_cloud,
        )
        out_dir = tmp_path / "output"

        config = PipelineConfig(
            preprocess=PreprocessConfig(
                downsample_target=200,
                sor_k=0,
            ),
            visualize=False,
        )

        result = register(src_path, tgt_path, config, out_dir)

        # Check outputs exist
        assert (out_dir / "aligned_source.glb").exists()
        assert (out_dir / "target.glb").exists()
        assert (out_dir / "merged.glb").exists()
        assert (out_dir / "results.h5").exists()

        # Check metrics present
        assert "chamfer" in result.metrics
        assert "rmse" in result.metrics
        assert result.metrics["rmse"] < 0.5

    def test_rigid_only(
        self, sphere_cloud, transformed_sphere, tmp_path,
    ):
        """Pipeline with rigid_only=True skips non-rigid."""
        src_path = _create_test_glb(
            tmp_path / "src.glb", transformed_sphere,
        )
        tgt_path = _create_test_glb(
            tmp_path / "tgt.glb", sphere_cloud,
        )

        config = PipelineConfig(
            preprocess=PreprocessConfig(
                downsample_target=200, sor_k=0,
            ),
            rigid_only=True,
            visualize=False,
        )

        result = register(
            src_path, tgt_path, config, tmp_path / "out",
        )
        assert result.nonrigid_result is None
        assert result.displacement_field is None