"""Tests for rigid and non-rigid CPD modules."""

import numpy as np
import pytest

from plant_3d.infrastructure.cpd.config import NonrigidCPDConfig, RigidCPDConfig
from plant_3d.infrastructure.cpd.cpd_nonrigid import nonrigid_cpd
from plant_3d.infrastructure.cpd.cpd_rigid import rigid_cpd
from plant_3d.infrastructure.cpd.metrics import rmse


class TestRigidCPD:
    def test_identity(self, sphere_cloud):
        """Identical clouds should produce near-identity."""
        result = rigid_cpd(sphere_cloud, sphere_cloud)
        np.testing.assert_allclose(
            result.rotation, np.eye(3), atol=0.1,
        )
        assert rmse(result.aligned, sphere_cloud) < 0.05

    def test_recovers_transform(
        self, sphere_cloud, transformed_sphere,
    ):
        """Should recover a known rigid transform."""
        config = RigidCPDConfig(max_iterations=200)
        result = rigid_cpd(
            transformed_sphere, sphere_cloud, config,
        )
        error = rmse(result.aligned, sphere_cloud)
        assert error < 0.15, f"RMSE too high: {error:.4f}"

    def test_result_shape(self, sphere_cloud):
        result = rigid_cpd(sphere_cloud, sphere_cloud)
        assert result.aligned.shape == sphere_cloud.shape
        assert result.rotation.shape == (3, 3)
        assert result.translation.shape == (3,)


class TestNonrigidCPD:
    def test_deformed_sphere(
        self, sphere_cloud, deformed_sphere,
    ):
        """Non-rigid should improve fit for deformed data."""
        config = NonrigidCPDConfig(max_iterations=100)
        result = nonrigid_cpd(
            sphere_cloud, deformed_sphere, config,
        )
        error = rmse(result.deformed, deformed_sphere)
        assert error < 0.2, f"RMSE too high: {error:.4f}"

    def test_displacement_field(
        self, sphere_cloud, deformed_sphere,
    ):
        """Displacement field should have correct shape."""
        result = nonrigid_cpd(sphere_cloud, deformed_sphere)
        assert result.displacement_field.shape == sphere_cloud.shape
        # Upper hemisphere should have positive z displacement
        upper = sphere_cloud[:, 2] > 0.3
        if upper.any():
            mean_z_disp = result.displacement_field[upper, 2].mean()
            assert mean_z_disp > 0, "Expected upward displacement"

    def test_result_fields(
        self, sphere_cloud, deformed_sphere,
    ):
        result = nonrigid_cpd(sphere_cloud, deformed_sphere)
        assert result.W.shape[1] == 3
        assert result.G.shape[0] == result.G.shape[1]
        assert result.iterations > 0