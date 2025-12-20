"""Rigid CPD registration step."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from pycpd import RigidRegistration

from plant_3d.infrastructure.cpd.config import RigidCPDConfig

logger = logging.getLogger(__name__)


@dataclass
class RigidResult:
    """Result of rigid CPD registration.
    """

    rotation: np.ndarray
    translation: np.ndarray
    scale: float
    aligned: np.ndarray
    iterations: int


def rigid_cpd(
    source: np.ndarray,
    target: np.ndarray,
    config: RigidCPDConfig | None = None,
    *,
    progress_callback: object | None = None,
) -> RigidResult:
    """Run rigid CPD.
    """
    if config is None:
        config = RigidCPDConfig()

    logger.info(
        "Rigid CPD: source=%d target=%d (w=%.2f, scale=%s)",
        len(source),
        len(target),
        config.w,
        config.scale,
    )

    reg = RigidRegistration(
        X=target.astype(np.float64),
        Y=source.astype(np.float64),
        w=config.w,
        max_iterations=config.max_iterations,
        tolerance=config.tolerance,
    )

    aligned, params = reg.register()

    s_val: float = float(params[0]) if config.scale else 1.0
    rotation: np.ndarray = np.asarray(params[1])
    translation: np.ndarray = np.asarray(params[2]).ravel()

    result = RigidResult(
        rotation=rotation,
        translation=translation,
        scale=s_val,
        aligned=np.asarray(aligned),
        iterations=reg.iteration,
    )

    logger.info(
        "Rigid CPD converged in %d iterations (scale=%.4f)",
        result.iterations,
        result.scale,
    )
    return result