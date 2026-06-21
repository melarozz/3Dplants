from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class LeafStemSegmentationConfig:
    stem_radius_quantile: float = 0.2
    stem_radius_scale: float = 1.1
    min_component_vertices: int = 0
    compute_face_labels: bool = True


def segment_leaf_stem(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: LeafStemSegmentationConfig | None = None,
) -> dict[str, np.ndarray | None]:
    cfg = config or LeafStemSegmentationConfig()
    # Heuristic:
    # 1) find connected vertex components,
    # 2) component with smallest median radius to global axis is stem,
    # 3) all other components are leaves.
    xy = vertices[:, :2]
    center = np.median(xy, axis=0)
    radius = np.linalg.norm(xy - center, axis=1)

    ii = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    jj = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    data = np.ones(ii.shape[0], dtype=np.uint8)
    graph = coo_matrix((data, (ii, jj)), shape=(vertices.shape[0], vertices.shape[0]))
    graph = graph + graph.T
    _, comp_ids = connected_components(graph.tocsr(), directed=False, return_labels=True)

    labels_vertex = np.ones(vertices.shape[0], dtype=np.uint8)
    best_comp = None
    best_score = -float("inf")
    for comp in np.unique(comp_ids):
        mask = comp_ids == comp
        if np.sum(mask) < max(cfg.min_component_vertices, 3):
            continue
        z_extent = float(vertices[mask, 2].max() - vertices[mask, 2].min())
        radial_score = float(
            np.quantile(radius[mask], cfg.stem_radius_quantile) * cfg.stem_radius_scale
        )
        score = z_extent - radial_score
        if score > best_score:
            best_score = score
            best_comp = comp
    if best_comp is not None:
        labels_vertex[comp_ids == best_comp] = 0  # 0=stem, 1=leaf

    labels_face = None
    if config is None or config.compute_face_labels:
        leaf_votes = labels_vertex[faces]
        labels_face = (np.mean(leaf_votes, axis=1) >= 0.5).astype(np.uint8)

    return {"labels_vertex": labels_vertex, "labels_face": labels_face}


__all__ = ["LeafStemSegmentationConfig", "segment_leaf_stem"]

