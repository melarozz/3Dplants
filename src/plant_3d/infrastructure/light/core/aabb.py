from __future__ import annotations

import numpy as np

from plant_3d.infrastructure.light.core.ray import Ray


class AABB:
    """
    Axis-Aligned Bounding Box.
    """

    def __init__(self, min_pt: np.ndarray, max_pt: np.ndarray) -> None:
        self.min: np.ndarray = np.asarray(min_pt, dtype=np.float64)
        self.max: np.ndarray = np.asarray(max_pt, dtype=np.float64)

    def intersect(self, ray: Ray, t_min_allowed: float = 0.0) -> bool:
        """
        Slab method for ray-AABB intersection.

        Returns True if intersection exists with t >= t_min_allowed.
        """
        dir_inv = 1.0 / (ray.d + 1e-20)

        tmin = (self.min - ray.o) * dir_inv
        tmax = (self.max - ray.o) * dir_inv

        t1 = np.minimum(tmin, tmax)
        t2 = np.maximum(tmin, tmax)

        t_enter = np.max(t1)
        t_exit = np.min(t2)

        return t_exit >= max(t_enter, t_min_allowed)