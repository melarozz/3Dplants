from __future__ import annotations

import numpy as np


def ray_triangle_intersect(
    ray_o: np.ndarray,
    ray_d: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    eps: float = 1e-8,
) -> tuple[float, float, float] | None:
    """
    Möller–Trumbore ray-triangle intersection.

    Returns
    -------
    (t, u, v) if hit, otherwise None
    """
    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(ray_d, edge2)
    a = np.dot(edge1, h)

    if -eps < a < eps:
        return None

    f = 1.0 / a
    s = ray_o - v0

    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = f * np.dot(ray_d, q)

    if v < 0.0 or (u + v) > 1.0:
        return None

    t = f * np.dot(edge2, q)

    if t > eps:
        return t, u, v

    return None