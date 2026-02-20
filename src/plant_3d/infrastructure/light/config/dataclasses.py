from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass
class LightConfig:
    type: Literal["point", "directional"]
    position: Tuple[float, float, float] | None = None
    direction: Tuple[float, float, float] | None = None
    intensity: float = 1.0


@dataclass
class SimulationConfig:
    epsilon: float = 1e-5
    shadow_attenuation: float = 0.1
    scale_to_height: float = 0.1