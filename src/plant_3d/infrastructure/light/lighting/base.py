from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Light(ABC):
    """
    Abstract light source.
    """

    @abstractmethod
    def direction(self, point: np.ndarray) -> np.ndarray:
        """
        Direction from point to light (normalized).
        """
        pass

    @abstractmethod
    def intensity(self, point: np.ndarray) -> float:
        """
        Light intensity at given point.
        """
        pass