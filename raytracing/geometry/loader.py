import numpy as np
import trimesh


def load_glb_to_triangles(filename):
    """
    Loads a GLB/GLTF file and returns vertices, faces, and the concatenated mesh object.
    """
    print(f"[Load] Loading: {filename}")
    scene = trimesh.load(filename, force='scene')
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh = scene
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    print(f"[Load] {len(vertices)} vertices, {len(faces)} triangles")
    return vertices, faces, mesh


def apply_rotation(vertices, rotation_matrix):
    """
    Apply a (3x3) rotation matrix to vertices (N x 3).
    """
    return vertices @ rotation_matrix.T
