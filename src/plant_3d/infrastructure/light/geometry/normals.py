from __future__ import annotations

import numpy as np


def compute_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """
    Compute per-vertex normals using area-weighted triangle normals.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # triangle normals (not normalized = weighted by area)
    tri_normals = np.cross(v1 - v0, v2 - v0)

    vert_normals = np.zeros_like(vertices)

    # accumulate
    for i, f in enumerate(faces):
        for vid in f:
            vert_normals[vid] += tri_normals[i]

    # normalize
    norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)

    # avoid division by zero
    vert_normals = np.divide(
        vert_normals,
        norms,
        where=norms > 0
    )

    # fallback for degenerate vertices
    zero_mask = (norms.squeeze() == 0)
    if np.any(zero_mask):
        vert_normals[zero_mask] = np.array([0.0, 0.0, 1.0])

    return vert_normals