"""Registration quality metrics."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def chamfer_distance(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Symmetric Chamfer distance between two point clouds.

    .. math::
        d_{CD} = \\frac{1}{|A|} \\sum_{a} \\min_b \\|a-b\\|^2
               + \\frac{1}{|B|} \\sum_{b} \\min_a \\|b-a\\|^2

    Parameters
    ----------
    a, b : np.ndarray
        Point clouds ``(M, 3)`` and ``(N, 3)``.

    Returns
    -------
    float
        Chamfer distance (sum of both directions).
    """
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)

    dists_a2b, _ = tree_b.query(a)
    dists_b2a, _ = tree_a.query(b)

    return float(
        (dists_a2b ** 2).mean() + (dists_b2a ** 2).mean(),
    )


def hausdorff_distance(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Symmetric Hausdorff distance.

    .. math::
        d_H = \\max\\bigl(
            \\max_a \\min_b \\|a-b\\|,\\;
            \\max_b \\min_a \\|b-a\\|
        \\bigr)

    Parameters
    ----------
    a, b : np.ndarray
        Point clouds ``(M, 3)`` and ``(N, 3)``.

    Returns
    -------
    float
        Hausdorff distance.
    """
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)

    dists_a2b, _ = tree_b.query(a)
    dists_b2a, _ = tree_a.query(b)

    return float(max(dists_a2b.max(), dists_b2a.max()))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root Mean Square Error of nearest-neighbour distances.

    For each point in *a*, finds the nearest point in *b* and
    computes RMSE of those distances.

    Parameters
    ----------
    a, b : np.ndarray
        Point clouds ``(M, 3)`` and ``(N, 3)``.

    Returns
    -------
    float
        RMSE value.
    """
    tree_b = cKDTree(b)
    dists, _ = tree_b.query(a)
    return float(np.sqrt((dists ** 2).mean()))


def inlier_ratio(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float = 0.05,
) -> float:
    """Fraction of source points within *threshold* of target.

    Parameters
    ----------
    a : np.ndarray
        Aligned source points ``(M, 3)``.
    b : np.ndarray
        Target points ``(N, 3)``.
    threshold : float
        Distance threshold.

    Returns
    -------
    float
        Ratio in ``[0, 1]``.
    """
    tree_b = cKDTree(b)
    dists, _ = tree_b.query(a)
    return float((dists < threshold).mean())


def compute_all_metrics(
    source: np.ndarray,
    target: np.ndarray,
    inlier_threshold: float = 0.05,
) -> dict[str, float]:
    """Compute all registration quality metrics.

    Parameters
    ----------
    source : np.ndarray
        Aligned source points ``(M, 3)``.
    target : np.ndarray
        Target points ``(N, 3)``.
    inlier_threshold : float
        Distance threshold for inlier ratio.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``chamfer``, ``hausdorff``,
        ``rmse``, ``inlier_ratio``.
    """
    return {
        "chamfer": chamfer_distance(source, target),
        "hausdorff": hausdorff_distance(source, target),
        "rmse": rmse(source, target),
        "inlier_ratio": inlier_ratio(
            source, target, inlier_threshold,
        ),
    }