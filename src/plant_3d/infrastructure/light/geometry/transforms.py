from __future__ import annotations

import numpy as np


def apply_rotation(
    vertices: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """
    Apply 3x3 rotation matrix to vertices.
    """
    return vertices @ rotation_matrix.T


def scale_to_height(
    vertices: np.ndarray,
    target_height: float,
) -> np.ndarray:
    """
    Scale mesh so its Z-height becomes target_height.

    Also shifts mesh so min Z = 0.
    """
    z_min = np.min(vertices[:, 2])
    z_max = np.max(vertices[:, 2])

    current_height = z_max - z_min

    if current_height <= 0:
        return vertices.copy()

    scale = target_height / current_height
    vertices_scaled = vertices * scale

    # shift to floor
    vertices_scaled[:, 2] -= np.min(vertices_scaled[:, 2])

    return vertices_scaled


def center_to_origin(vertices: np.ndarray) -> np.ndarray:
    """
    Center mesh at origin.
    """
    center = np.mean(vertices, axis=0)
    return vertices - center


def compute_bbox(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute bounding box (min, max).
    """
    return np.min(vertices, axis=0), np.max(vertices, axis=0)