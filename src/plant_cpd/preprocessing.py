"""Point-cloud pre-processing: downsample, filter, normalise."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from plant_cpd.config import PreprocessConfig

logger = logging.getLogger(__name__)


def center(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate points so that the centroid is at the origin.

    Parameters
    ----------
    points : np.ndarray
        Input points ``(N, 3)``.

    Returns
    -------
    centered : np.ndarray
        Centered points ``(N, 3)``.
    centroid : np.ndarray
        Original centroid ``(3,)``.
    """
    centroid = points.mean(axis=0)
    return points - centroid, centroid


def normalize_scale(
    points: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Scale points so that the max extent equals 1.

    Parameters
    ----------
    points : np.ndarray
        Input points ``(N, 3)``.

    Returns
    -------
    scaled : np.ndarray
        Normalised points ``(N, 3)``.
    scale_factor : float
        The factor that was applied (``1 / extent``).
    """
    extent = points.max() - points.min()
    if extent < 1e-12:
        logger.warning("Point cloud has near-zero extent")
        return points.copy(), 1.0
    scale_factor = 1.0 / extent
    return points * scale_factor, scale_factor


def voxel_downsample(
    points: np.ndarray,
    target_n: int,
) -> np.ndarray:
    """Downsample a point cloud using a voxel grid.

    Automatically computes the voxel size so that approximately
    *target_n* points remain.

    Parameters
    ----------
    points : np.ndarray
        Input points ``(N, 3)``.
    target_n : int
        Desired number of output points.

    Returns
    -------
    np.ndarray
        Downsampled points ``(~target_n, 3)``.
    """
    n = len(points)
    if n <= target_n:
        logger.debug("Skipping downsample: %d <= %d", n, target_n)
        return points.copy()

    # Estimate voxel size from bounding box and target count
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    volume = np.prod(bbox_size)
    voxel_size = (volume / target_n) ** (1.0 / 3.0)

    # Assign each point to a voxel
    voxel_indices = np.floor(
        (points - bbox_min) / max(voxel_size, 1e-12),
    ).astype(np.int64)

    # Use structured array for unique voxel identification
    _, unique_idx = np.unique(
        voxel_indices, axis=0, return_index=True,
    )

    downsampled = points[np.sort(unique_idx)]
    logger.info(
        "Voxel downsample: %d → %d (voxel=%.4f)",
        n,
        len(downsampled),
        voxel_size,
    )
    return downsampled


def statistical_outlier_removal(
    points: np.ndarray,
    k: int,
    std_ratio: float,
) -> np.ndarray:
    """Remove statistical outliers via k-NN distance analysis.

    A point is considered an outlier if its mean distance to
    *k* nearest neighbours exceeds ``mean + std_ratio * std``
    of all such distances.

    Parameters
    ----------
    points : np.ndarray
        Input points ``(N, 3)``.
    k : int
        Number of neighbours.
    std_ratio : float
        Standard-deviation multiplier.

    Returns
    -------
    np.ndarray
        Filtered points.
    """
    if k <= 0:
        return points.copy()

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k + 1)  # includes self
    mean_dists = distances[:, 1:].mean(axis=1)

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std

    mask = mean_dists < threshold
    filtered = points[mask]
    n_removed = len(points) - len(filtered)
    logger.info(
        "SOR: removed %d / %d outliers (threshold=%.4f)",
        n_removed,
        len(points),
        threshold,
    )
    return filtered


def preprocess(
    points: np.ndarray,
    config: PreprocessConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run the full pre-processing pipeline.

    Steps (in order):
    1. Center (optional)
    2. Normalize scale (optional)
    3. Voxel downsample (optional)
    4. Statistical Outlier Removal (optional)

    Parameters
    ----------
    points : np.ndarray
        Raw vertices ``(N, 3)``.
    config : PreprocessConfig
        Pre-processing settings.

    Returns
    -------
    processed : np.ndarray
        Pre-processed points.
    meta : dict[str, object]
        Metadata dict with ``centroid``, ``scale_factor``,
        ``n_original``, ``n_processed``.
    """
    meta: dict[str, object] = {"n_original": len(points)}
    result = points.astype(np.float64).copy()

    if config.center:
        result, centroid = center(result)
        meta["centroid"] = centroid
    else:
        meta["centroid"] = np.zeros(3)

    if config.normalize_scale:
        result, scale_factor = normalize_scale(result)
        meta["scale_factor"] = scale_factor
    else:
        meta["scale_factor"] = 1.0

    if config.downsample_target is not None:
        result = voxel_downsample(result, config.downsample_target)

    if config.sor_k > 0:
        result = statistical_outlier_removal(
            result, config.sor_k, config.sor_std_ratio,
        )

    meta["n_processed"] = len(result)
    logger.info(
        "Preprocessing complete: %d → %d points",
        meta["n_original"],
        meta["n_processed"],
    )
    return result, meta