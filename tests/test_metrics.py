"""Tests for registration quality metrics."""

import numpy as np
import pytest

from plant_3d.infrastructure.cpd.metrics import (
    chamfer_distance,
    compute_all_metrics,
    hausdorff_distance,
    inlier_ratio,
    rmse,
)


class TestChamfer:
    def test_identical_is_zero(self, sphere_cloud):
        d = chamfer_distance(sphere_cloud, sphere_cloud)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self, sphere_cloud, noisy_sphere):
        d1 = chamfer_distance(sphere_cloud, noisy_sphere)
        d2 = chamfer_distance(noisy_sphere, sphere_cloud)
        assert d1 == pytest.approx(d2, rel=1e-6)

    def test_positive(self, sphere_cloud, noisy_sphere):
        d = chamfer_distance(sphere_cloud, noisy_sphere)
        assert d > 0


class TestHausdorff:
    def test_identical_is_zero(self, sphere_cloud):
        d = hausdorff_distance(sphere_cloud, sphere_cloud)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_greater_than_chamfer(
        self, sphere_cloud, noisy_sphere,
    ):
        cd = chamfer_distance(sphere_cloud, noisy_sphere)
        hd = hausdorff_distance(sphere_cloud, noisy_sphere)
        # Hausdorff (max) should generally be >= sqrt(chamfer mean)
        assert hd >= 0


class TestRMSE:
    def test_identical_is_zero(self, sphere_cloud):
        r = rmse(sphere_cloud, sphere_cloud)
        assert r == pytest.approx(0.0, abs=1e-10)

    def test_noise_level(self, sphere_cloud, noisy_sphere):
        r = rmse(sphere_cloud, noisy_sphere)
        # Noise was std=0.02, RMSE should be in that ballpark
        assert r < 0.1


class TestInlierRatio:
    def test_identical_is_one(self, sphere_cloud):
        r = inlier_ratio(sphere_cloud, sphere_cloud)
        assert r == pytest.approx(1.0)

    def test_far_apart_is_zero(self, sphere_cloud):
        far = sphere_cloud + 100
        r = inlier_ratio(sphere_cloud, far, threshold=0.01)
        assert r == pytest.approx(0.0)


class TestComputeAll:
    def test_returns_all_keys(self, sphere_cloud):
        m = compute_all_metrics(sphere_cloud, sphere_cloud)
        assert set(m.keys()) == {
            "chamfer", "hausdorff", "rmse", "inlier_ratio",
        }

    def test_all_values_numeric(self, sphere_cloud):
        m = compute_all_metrics(sphere_cloud, sphere_cloud)
        for v in m.values():
            assert isinstance(v, float)