"""Shared pytest fixtures for plant-cpd tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(seed=42)


@pytest.fixture()
def sphere_cloud(rng: np.random.Generator) -> np.ndarray:
    """Unit sphere point cloud (500 points)."""
    n = 500
    phi = rng.uniform(0, 2 * np.pi, n)
    cos_theta = rng.uniform(-1, 1, n)
    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


@pytest.fixture()
def cylinder_cloud(rng: np.random.Generator) -> np.ndarray:
    """Cylinder point cloud (plant-stem-like, 500 points)."""
    n = 500
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-1, 1, n)
    r = 0.3
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])


@pytest.fixture()
def rigid_transform():
    """A known rigid transform (rotation + translation)."""
    rot = Rotation.from_euler("xyz", [15, 25, 10], degrees=True)
    r_mat = rot.as_matrix()
    t_vec = np.array([0.5, -0.3, 0.2])
    return r_mat, t_vec


@pytest.fixture()
def transformed_sphere(
    sphere_cloud: np.ndarray,
    rigid_transform: tuple,
) -> np.ndarray:
    """Sphere with a known rigid transform applied."""
    r_mat, t_vec = rigid_transform
    return (sphere_cloud @ r_mat.T) + t_vec


@pytest.fixture()
def noisy_sphere(
    sphere_cloud: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sphere with additive Gaussian noise."""
    noise = rng.normal(scale=0.02, size=sphere_cloud.shape)
    return sphere_cloud + noise


@pytest.fixture()
def deformed_sphere(
    sphere_cloud: np.ndarray,
) -> np.ndarray:
    """Sphere with a smooth non-rigid deformation applied."""
    deformed = sphere_cloud.copy()
    # Stretch the upper hemisphere
    mask = deformed[:, 2] > 0
    deformed[mask, 2] *= 1.3
    return deformed


@pytest.fixture()
def tmp_h5(tmp_path):
    """Temporary HDF5 file path."""
    return tmp_path / "test_results.h5"