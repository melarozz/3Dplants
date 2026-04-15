import numpy as np

from plant_3d.infrastructure.light.core.bvh import build_bvh, traverse_bvh
from plant_3d.infrastructure.light.core.ray import Ray


def test_bvh_basic():
    vertices = np.array([
        [0, 0, 5],
        [1, 0, 5],
        [0, 1, 5],
    ])

    faces = np.array([[0, 1, 2]])

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    centers = (v0 + v1 + v2) / 3
    tri_min = np.minimum(np.minimum(v0, v1), v2)
    tri_max = np.maximum(np.maximum(v0, v1), v2)

    bvh = build_bvh(centers, tri_min, tri_max, [0])

    ray = Ray(np.array([0.1, 0.1, 0]), np.array([0, 0, 1]))

    hit = traverse_bvh(bvh, ray, v0, v1, v2)

    assert hit is not None