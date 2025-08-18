import numpy as np


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n > 0.0:
        return v / n
    return v


class Ray:
    """
    Simple Ray container with origin (o) and normalized direction (d).
    Use Ray(origin, direction) to construct; direction will be normalized automatically.
    """

    def __init__(self, origin, direction):
        self.o = np.asarray(origin, dtype=np.float64)
        self.d = normalize(direction)

    def point_at(self, t: float):
        return self.o + self.d * t
