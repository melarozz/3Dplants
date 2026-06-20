from __future__ import annotations

import numpy as np
import trimesh

from plant_light_sim.utils.logging import get_logger

logger = get_logger(__name__)


def load_glb_to_triangles(
    filename: str,
) -> tuple[np.ndarray, np.ndarray, trimesh.Trimesh]:
    """
    Load GLB/GLTF file and return vertices, faces and merged mesh.

    If file contains a scene, all geometries are concatenated.

    Returns
    -------
    vertices : (N, 3)
    faces : (M, 3)
    mesh : trimesh.Trimesh
    """
    logger.info(f"Loading mesh: {filename}")

    scene = trimesh.load(filename, force="scene")

    if isinstance(scene, trimesh.Scene):
        if not scene.geometry:
            raise ValueError("Scene contains no geometry")

        mesh = trimesh.util.concatenate(
            [g for g in scene.geometry.values()]
        )
    else:
        mesh = scene

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    logger.info(f"Loaded {len(vertices)} vertices, {len(faces)} triangles")

    return vertices, faces, mesh