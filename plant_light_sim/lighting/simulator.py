from __future__ import annotations

import numpy as np
from plant_light_sim.core.ray import Ray
from plant_light_sim.core.bvh import traverse_bvh
from plant_light_sim.geometry.normals import compute_vertex_normals
from plant_light_sim.utils.logging import get_logger

logger = get_logger(__name__)


class LightSimulator:
    """
    Compute PPFD per vertex for a plant mesh given a light source
    """

    def __init__(self, vertices, faces, bvh, v0, v1, v2):
        self.vertices = vertices
        self.faces = faces
        self.bvh = bvh
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    def compute_vertex_heatmap(
        self, light, shadow_attenuation: float = 0.1, epsilon: float = 1e-5
    ) -> np.ndarray:
        """
        Returns PPFD per vertex (µmol/m²/s)
        """
        normals = compute_vertex_normals(self.vertices, self.faces)
        n = len(self.vertices)
        ppfd_values = np.zeros(n, dtype=np.float64)

        for i, (p, nrm) in enumerate(zip(self.vertices, normals)):
            to_light = light.direction_to_light(p)
            dist = np.linalg.norm(to_light)

            if dist == 0:
                continue

            dir_norm = to_light / dist
            lambert = max(0.0, np.dot(nrm, dir_norm))

            base = light.compute_intensity(p)
            value = base * lambert

            # shadow
            ray = Ray(p + nrm * epsilon, dir_norm)
            hit = traverse_bvh(self.bvh, ray, self.v0, self.v1, self.v2)
            if hit and hit[0] < dist:
                value *= shadow_attenuation

            ppfd_values[i] = value

            if i % 10000 == 0:
                logger.info(f"{i}/{n} vertices processed")

        logger.info("PPFD computation finished")
        return ppfd_values