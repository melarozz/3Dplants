from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LabeledPointCloud:
    points: np.ndarray
    labels: np.ndarray
    colors: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.points.shape[0] != self.labels.shape[0]:
            msg = f"Labels count mismatch: {self.labels.shape[0]} != {self.points.shape[0]}"
            raise ValueError(msg)
