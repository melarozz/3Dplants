import numpy as np
from plant_light_sim.core.intersection import ray_triangle_intersect


def test_intersection_hit():
    ray_o = np.array([0, 0, 0])
    ray_d = np.array([0, 0, 1])

    v0 = np.array([-1, -1, 5])
    v1 = np.array([1, -1, 5])
    v2 = np.array([0, 1, 5])

    res = ray_triangle_intersect(ray_o, ray_d, v0, v1, v2)

    assert res is not None