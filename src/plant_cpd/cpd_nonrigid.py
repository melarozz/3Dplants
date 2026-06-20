"""Non-rigid CPD registration step."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from pycpd import DeformableRegistration

from plant_cpd.config import NonrigidCPDConfig

logger = logging.getLogger(__name__)


@dataclass
class NonrigidResult:
    """Result of non-rigid CPD registration.

    Attributes
    ----------
    W : np.ndarray
        Weight matrix ``(M, 3)``.
    G : np.ndarray
        Gaussian affinity kernel ``(M, M)``.
    deformed : np.ndarray
        Deformed source points ``(M, 3)``.
    displacement_field : np.ndarray
        Per-point displacement vectors ``(M, 3)``
        (``deformed - source``).
    iterations : int
        Number of EM iterations performed.
    """

    W: np.ndarray
    G: np.ndarray
    deformed: np.ndarray
    displacement_field: np.ndarray
    iterations: int


def nonrigid_cpd(
    source: np.ndarray,
    target: np.ndarray,
    config: NonrigidCPDConfig | None = None,
) -> NonrigidResult:
    """Run non-rigid CPD to deform *source* onto *target*.

    Should be called **after** rigid alignment for best results.

    Parameters
    ----------
    source : np.ndarray
        Source point cloud ``(M, 3)`` (ideally rigid-aligned).
    target : np.ndarray
        Target (fixed) point cloud ``(N, 3)``.
    config : NonrigidCPDConfig | None
        Algorithm parameters.  Uses defaults if ``None``.

    Returns
    -------
    NonrigidResult
        Deformation result including displacement field.
    """
    if config is None:
        config = NonrigidCPDConfig()

    logger.info(
        "Non-rigid CPD: source=%d target=%d "
        "(beta=%.1f, lambda=%.1f, w=%.2f)",
        len(source),
        len(target),
        config.beta,
        config.lmbda,
        config.w,
    )

    reg = DeformableRegistration(
        X=target.astype(np.float64),
        Y=source.astype(np.float64),
        w=config.w,
        max_iterations=config.max_iterations,
        tolerance=config.tolerance,
        alpha=config.lmbda,  # pycpd uses 'alpha' for λ
        beta=config.beta,
    )

    deformed, _ = reg.register()
    deformed = np.asarray(deformed)

    displacement = deformed - source.astype(np.float64)

    result = NonrigidResult(
        W=np.asarray(reg.W),
        G=np.asarray(reg.G),
        deformed=deformed,
        displacement_field=displacement,
        iterations=reg.iteration,
    )

    logger.info(
        "Non-rigid CPD converged in %d iterations",
        result.iterations,
    )
    logger.info(
        "  displacement: mean=%.4f max=%.4f",
        float(np.linalg.norm(displacement, axis=1).mean()),
        float(np.linalg.norm(displacement, axis=1).max()),
    )
    return result