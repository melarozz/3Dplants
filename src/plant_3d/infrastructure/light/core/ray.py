from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector. If norm is zero, return the original vector.
    """
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n > 0.0:
        return v / n
    return v


class Ray:
    """
    Ray with origin and normalized direction.
    """

    def __init__(self, origin: np.ndarray, direction: np.ndarray) -> None:
        self.o: np.ndarray = np.asarray(origin, dtype=np.float64)
        self.d: np.ndarray = normalize(direction)

    def point_at(self, t: float) -> np.ndarray:
        """
        Compute point along ray: o + t * d
        """
        return self.o + self.d * t