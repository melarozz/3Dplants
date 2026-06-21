from __future__ import annotations
import os
import numpy as np
import trimesh
import matplotlib.cm as cm
from plant_3d.infrastructure.light.utils.logging import get_logger
logger = get_logger(__name__)

def tone_map_relative(values):
    v = values.copy()
    (p1, p99) = np.percentile(v, [
        1,
        99])
    v = np.clip((v - p1) / ((p99 - p1) + 1e-08), 0, 1)
    v = v ** 0.5
    v *= 3
    return np.clip(v, 0, 1)


def tone_map_absolute(values, max_ppfd = 1000):
    '''
    Absolute mapping
    '''
    v = values / max_ppfd
    return np.clip(v, 0, 1)


def save_colored_glb(vertices, faces, light_vals, output_path, light_pos = None, add_lamp = True, mode = 'relative'):
    os.makedirs(os.path.dirname(output_path), exist_ok = True)
    if mode == 'absolute':
        lv = tone_map_absolute(light_vals)
    else:
        lv = tone_map_relative(light_vals)
    colors = (cm.plasma(lv) * 255).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices = vertices, faces = faces, process = False)
    mesh.visual.vertex_colors = colors
    meshes = [
        mesh]
    if add_lamp and light_pos:
        bbox_diag = np.linalg.norm(np.max(vertices, axis = 0) - np.min(vertices, axis = 0))
        radius = max(0.005, bbox_diag * 0.02)
        lamp = trimesh.creation.icosphere(subdivisions = 3, radius = radius)
        lamp.visual.vertex_colors = np.tile(np.array([
            255,
            230,
            0,
            255], dtype = np.uint8), (len(lamp.vertices), 1))
        lamp.apply_translation(light_pos)
        meshes.append(lamp)
    combined = trimesh.util.concatenate(meshes)
    logger.info(f'''Exporting GLB: {output_path}''')
    combined.export(output_path)

