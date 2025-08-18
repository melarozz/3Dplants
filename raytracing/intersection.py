import numpy as np

def ray_triangle_intersect(ray_o, ray_d, v0, v1, v2, eps=1e-8):
    """Möller–Trumbore ray-triangle intersection."""
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
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * np.dot(edge2, q)
    if t > eps:
        return t, u, v
    return None
