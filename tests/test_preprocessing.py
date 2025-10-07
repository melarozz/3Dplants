"""Tests for the preprocessing module."""

import numpy as np
import pytest

from plant_cpd.config import PreprocessConfig
from plant_cpd.preprocessing import (
    center,
    normalize_scale,
    preprocess,
    statistical_outlier_removal,
    voxel_downsample,
)


class TestCenter:
    def test_centroid_at_origin(self, sphere_cloud):
        centered, centroid = center(sphere_cloud)
        np.testing.assert_allclose(
            centered.mean(axis=0), [0, 0, 0], atol=1e-10,
        )

    def test_centroid_returned(self, sphere_cloud):
        _, centroid = center(sphere_cloud)
        expected = sphere_cloud.mean(axis=0)
        np.testing.assert_allclose(centroid, expected)


class TestNormalizeScale:
    def test_max_extent_is_one(self, sphere_cloud):
        scaled, _ = normalize_scale(sphere_cloud)
        extent = scaled.max() - scaled.min()
        assert abs(extent - 1.0) < 1e-10

    def test_scale_factor_positive(self, sphere_cloud):
        _, factor = normalize_scale(sphere_cloud)
        assert factor > 0


class TestVoxelDownsample:
    def test_reduces_points(self, sphere_cloud):
        result = voxel_downsample(sphere_cloud, target_n=100)
        assert len(result) <= len(sphere_cloud)
        assert len(result) > 0

    def test_no_op_when_small(self):
        pts = np.random.randn(50, 3)
        result = voxel_downsample(pts, target_n=100)
        assert len(result) == 50


class TestSOR:
    def test_removes_outliers(self, sphere_cloud, rng):
        # Add 10 extreme outliers
        outliers = rng.uniform(10, 20, (10, 3))
        noisy = np.vstack([sphere_cloud, outliers])
        filtered = statistical_outlier_removal(noisy, k=20, std_ratio=2.0)
        assert len(filtered) < len(noisy)
        assert len(filtered) >= len(sphere_cloud) - 5

    def test_skip_when_k_zero(self, sphere_cloud):
        result = statistical_outlier_removal(
            sphere_cloud, k=0, std_ratio=2.0,
        )
        assert len(result) == len(sphere_cloud)


class TestPreprocess:
    def test_full_pipeline(self, sphere_cloud):
        config = PreprocessConfig(downsample_target=200)
        result, meta = preprocess(sphere_cloud, config)
        assert result.shape[1] == 3
        assert meta["n_original"] == 500
        assert meta["n_processed"] <= 500

    def test_no_downsample(self, sphere_cloud):
        config = PreprocessConfig(downsample_target=None, sor_k=0)
        result, _ = preprocess(sphere_cloud, config)
        assert len(result) == 500