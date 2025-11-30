"""plant-cpd: CPD-based registration of 3D plant models."""

from plant_cpd.config import (
    NonrigidCPDConfig,
    PipelineConfig,
    PreprocessConfig,
    RigidCPDConfig,
)
from plant_cpd.pipeline import register

__version__ = "0.1.0"

__all__ = [
    "register",
    "PipelineConfig",
    "PreprocessConfig",
    "RigidCPDConfig",
    "NonrigidCPDConfig",
    "__version__",
]