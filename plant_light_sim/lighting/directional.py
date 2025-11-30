from __future__ import annotations

import numpy as np


class DirectionalLight:
    """
    Directional light (like sunlight), intensity in µmol/m²/s
    """

    def __init__(self, direction: np.ndarray, intensity: float = 200.0):
        d = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(d)
        if norm == 0:
            raise ValueError("Direction vector cannot be zero")
        self.direction_vec = d / norm
        self.intensity_val = float(intensity)

    def direction_to_light(self, point: np.ndarray) -> np.ndarray:
        # Directional light comes FROM the direction
        return -self.direction_vec

    def compute_intensity(self, point: np.ndarray) -> float:
        return self.intensity_val