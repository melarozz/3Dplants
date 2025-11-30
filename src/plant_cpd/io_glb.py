"""GLB file I/O: load point clouds, export aligned meshes."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


def load_glb(path: str | Path) -> np.ndarray:
    """Load a GLB file and return vertices as an (N, 3) float64 array.

    Handles both single-mesh files and multi-mesh scenes
    (common in Hunyuan3D output) by concatenating all mesh
    vertices.

    Parameters
    ----------
    path : str | Path
        Path to the ``.glb`` file.

    Returns
    -------
    np.ndarray
        Vertex positions with shape ``(N, 3)``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file contains no geometry.
    """
    path = Path(path)
    if not path.exists():
        msg = f"GLB file not found: {path}"
        raise FileNotFoundError(msg)

    scene = trimesh.load(str(path), force="scene")

    vertices_list: list[np.ndarray] = []
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            logger.debug(
                "Mesh '%s': %d vertices", name, len(geom.vertices),
            )
            vertices_list.append(np.asarray(geom.vertices))

    if not vertices_list:
        msg = f"No meshes found in {path}"
        raise ValueError(msg)

    points = np.concatenate(vertices_list, axis=0).astype(np.float64)
    logger.info(
        "Loaded %d vertices from '%s' (%d meshes)",
        len(points),
        path.name,
        len(vertices_list),
    )
    return points


def load_glb_scene(path: str | Path) -> trimesh.Scene:
    """Load a GLB file as a full trimesh Scene (preserving faces).

    Parameters
    ----------
    path : str | Path
        Path to the ``.glb`` file.

    Returns
    -------
    trimesh.Scene
        The loaded scene object.
    """
    return trimesh.load(str(path), force="scene")


def export_glb(
    points: np.ndarray,
    path: str | Path,
    *,
    color: tuple[int, int, int, int] = (100, 200, 100, 255),
) -> Path:
    """Export a point cloud as a GLB file (point-cloud mesh).

    Creates a trimesh PointCloud, wraps it into a Scene, and
    exports as GLB.

    Parameters
    ----------
    points : np.ndarray
        Vertex positions ``(N, 3)``.
    path : str | Path
        Destination file path.
    color : tuple[int, int, int, int]
        RGBA colour for all points.

    Returns
    -------
    Path
        The written file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    colors = np.tile(color, (len(points), 1)).astype(np.uint8)
    cloud = trimesh.PointCloud(vertices=points, colors=colors)
    scene = trimesh.Scene([cloud])
    scene.export(str(path), file_type="glb")
    logger.info("Exported %d points → '%s'", len(points), path)
    return path


def export_merged_glb(
    source_points: np.ndarray,
    target_points: np.ndarray,
    path: str | Path,
) -> Path:
    """Export two point clouds as a single merged GLB scene.

    Source is coloured green, target is coloured blue.

    Parameters
    ----------
    source_points : np.ndarray
        Aligned source vertices ``(M, 3)``.
    target_points : np.ndarray
        Target vertices ``(N, 3)``.
    path : str | Path
        Destination path.

    Returns
    -------
    Path
        The written file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    src_colors = np.tile(
        [80, 200, 120, 255], (len(source_points), 1),
    ).astype(np.uint8)
    tgt_colors = np.tile(
        [100, 149, 237, 255], (len(target_points), 1),
    ).astype(np.uint8)

    src_cloud = trimesh.PointCloud(
        vertices=source_points, colors=src_colors,
    )
    tgt_cloud = trimesh.PointCloud(
        vertices=target_points, colors=tgt_colors,
    )
    scene = trimesh.Scene([src_cloud, tgt_cloud])
    scene.export(str(path), file_type="glb")
    logger.info("Exported merged scene → '%s'", path)
    return path