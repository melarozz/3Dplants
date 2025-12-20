"""Configuration dataclasses for the plant-cpd pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings for point-cloud pre-processing.
    """

    downsample_target: int | None = 5000
    sor_k: int = 20
    sor_std_ratio: float = 2.0
    center: bool = True
    normalize_scale: bool = True


@dataclass(frozen=True)
class RigidCPDConfig:
    """Parameters for the rigid CPD step.
    """

    w: float = 0.1
    max_iterations: int = 150
    tolerance: float = 1e-6
    scale: bool = True


@dataclass(frozen=True)
class NonrigidCPDConfig:
    """Parameters for the non-rigid CPD step.
    """

    w: float = 0.1
    max_iterations: int = 150
    tolerance: float = 1e-6
    beta: float = 3.0
    lmbda: float = 2.0


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs.
    """

    preprocess: PreprocessConfig = field(
        default_factory=PreprocessConfig,
    )
    rigid: RigidCPDConfig = field(default_factory=RigidCPDConfig)
    nonrigid: NonrigidCPDConfig = field(
        default_factory=NonrigidCPDConfig,
    )
    rigid_only: bool = False
    save_hdf5: bool = True
    hdf5_filename: str = "results.h5"
    visualize: bool = False