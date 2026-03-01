from __future__ import annotations

import numpy as np


class PointLight:
    """
    Physically-inspired point light.
    """

    def __init__(self, position: np.ndarray, power: float = 20.0):
        self.position = np.asarray(position, dtype=np.float64)
        self.power = float(power)

    def compute_intensity(self, point: np.ndarray) -> float:
        """
        Compute PPFD at a vertex using inverse-square law
        """
        d = np.linalg.norm(self.position - point)
        if d < 1e-6:
            return 0.0
        irradiance = self.power / (4 * np.pi * d * d)
        return irradiance

    def direction_to_light(self, point: np.ndarray) -> np.ndarray:
        return self.position - point