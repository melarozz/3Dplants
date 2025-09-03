"""Configuration dataclasses for the plant-cpd pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings for point-cloud pre-processing.

    Attributes:
        downsample_target: Target number of points after voxel
            downsampling.  ``None`` disables downsampling.
        sor_k: Number of neighbours for Statistical Outlier
            Removal.  Set to ``0`` to skip SOR.
        sor_std_ratio: Standard-deviation multiplier for SOR
            threshold.
        center: Whether to translate the centroid to the origin.
        normalize_scale: Whether to scale the cloud so that the
            max extent equals 1.
    """

    downsample_target: int | None = 5000
    sor_k: int = 20
    sor_std_ratio: float = 2.0
    center: bool = True
    normalize_scale: bool = True


@dataclass(frozen=True)
class RigidCPDConfig:
    """Parameters for the rigid CPD step.

    Attributes:
        w: Weight of the uniform distribution component
            (noise / outlier ratio).  Range ``[0, 1)``.
        max_iterations: Maximum EM iterations.
        tolerance: Convergence threshold on the log-likelihood.
        scale: Whether to estimate an isotropic scale factor.
    """

    w: float = 0.1
    max_iterations: int = 150
    tolerance: float = 1e-6
    scale: bool = True


@dataclass(frozen=True)
class NonrigidCPDConfig:
    """Parameters for the non-rigid CPD step.

    Attributes:
        w: Outlier weight.
        max_iterations: Maximum EM iterations.
        tolerance: Convergence threshold.
        beta: Width of the Gaussian kernel (smoothness).
            Larger values produce smoother deformations.
        lmbda: Trade-off between fit and regularisation.
            Larger values penalise large deformations.
    """

    w: float = 0.1
    max_iterations: int = 150
    tolerance: float = 1e-6
    beta: float = 3.0
    lmbda: float = 2.0


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs.

    Attributes:
        preprocess: Pre-processing settings.
        rigid: Rigid CPD settings.
        nonrigid: Non-rigid CPD settings.
        rigid_only: If ``True``, skip the non-rigid step.
        save_hdf5: If ``True``, persist every intermediate
            artefact to an HDF5 file.
        hdf5_filename: Name of the HDF5 results file inside
            *output_dir*.
        visualize: If ``True``, open a PyVista window after
            registration.
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