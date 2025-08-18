import numpy as np
import time
from ..ray import Ray, normalize
from ..bvh import build_bvh, traverse_bvh


class Lighting:
    """
    Lighting utilities for computing a per-vertex light heatmap from a single point light.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)

        # precompute triangle data
        self.v0 = self.vertices[self.faces[:, 0]]
        self.v1 = self.vertices[self.faces[:, 1]]
        self.v2 = self.vertices[self.faces[:, 2]]
        tri_centers = (self.v0 + self.v1 + self.v2) / 3.0
        tri_min = np.minimum(np.minimum(self.v0, self.v1), self.v2)
        tri_max = np.maximum(np.maximum(self.v0, self.v1), self.v2)

        self.tri_centers = tri_centers
        self.tri_min = tri_min
        self.tri_max = tri_max
        self.tri_indices = np.arange(len(self.faces))

        # build BVH
        self.bvh_root = build_bvh(self.tri_centers, self.tri_min, self.tri_max, self.tri_indices)

    def compute_vertex_normals(self):
        """
        Compute per-vertex normals by area-weighted averaging of triangle normals.
        Returns normalized normals array shaped (n_vertices, 3).
        """
        # triangle normals (unnormalized) using cross product
        tri_normals = np.cross(self.v1 - self.v0, self.v2 - self.v0)
        # avoid zero normals
        tri_normals = np.array([normalize(n) if np.linalg.norm(n) > 0 else np.zeros(3) for n in tri_normals])

        vert_normals = np.zeros_like(self.vertices)
        counts = np.zeros(len(self.vertices), dtype=np.int64)

        for i, f in enumerate(self.faces):
            for vid in f:
                vert_normals[vid] += tri_normals[i]
                counts[vid] += 1

        # avoid division by zero; only divide where count > 0
        nonzero = counts > 0
        vert_normals[nonzero] = vert_normals[nonzero] / counts[nonzero][:, None]
        vert_normals[~nonzero] = np.array([0.0, 0.0, 1.0])  # fallback normal

        vert_normals = np.array([normalize(n) for n in vert_normals])
        return vert_normals

    def compute_light_heatmap(self, light_pos, shadow_attenuation=0.2, epsilon=1e-5, log_interval=0.05):
        """
        Casts one ray per vertex toward the light and returns per-vertex illumination (Lambertian cosine)
        possibly attenuated by hard shadows (simple binary test).
        """
        light_pos = np.asarray(light_pos, dtype=np.float64)
        print("[Heatmap] Preparing geometry and normals...")
        vert_normals = self.compute_vertex_normals()

        print("[Heatmap] Casting rays...")
        total_vertices = len(self.vertices)
        light_values = np.zeros(total_vertices, dtype=np.float64)

        start = time.time()
        next_log = log_interval
        for vid, (p, nrm) in enumerate(zip(self.vertices, vert_normals)):
            # compute direction to light
            to_light = light_pos - p
            dist_to_light = np.linalg.norm(to_light)
            if dist_to_light == 0:
                # light at the vertex -> full illumination
                light_values[vid] = 1.0
                continue

            light_dir = to_light / dist_to_light
            lam = max(0.0, np.dot(nrm, light_dir))

            # offset origin a little along normal to avoid self-intersection
            shadow_ray = Ray(p + nrm * epsilon, light_dir)
            hit = traverse_bvh(self.bvh_root, shadow_ray, self.v0, self.v1, self.v2)
            if hit and hit[0] < dist_to_light:
                lam *= shadow_attenuation

            light_values[vid] = lam

            # logging
            progress = (vid + 1) / total_vertices
            if progress >= next_log:
                print(f"[Heatmap] {progress:.0%} of rays processed...")
                next_log += log_interval

        print(f"[Heatmap] Done in {time.time() - start:.2f}s")
        return light_values
