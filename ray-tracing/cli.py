import argparse
import numpy as np
from .geometry import load_glb_to_triangles, apply_rotation
from .lighting import Lighting
from .visualization import visualize


def main():
    parser = argparse.ArgumentParser(description="GLB Light Heatmap Visualizer")
    parser.add_argument("glb_file", help="Path to GLB/GLTF/GLB file")
    parser.add_argument("--light", nargs=3, type=float, default=[0.0, 0.0, 1.0],
                        help="Light position (x y z)")
    parser.add_argument("--rotation", nargs=9, type=float,
                        help="Optional 3x3 rotation matrix in row-major order (9 numbers)")
    args = parser.parse_args()

    vertices, faces, mesh = load_glb_to_triangles(args.glb_file)

    if args.rotation:
        rotation_matrix = np.array(args.rotation, dtype=np.float64).reshape((3, 3))
    else:
        # default rotation
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=np.float64)

    vertices = apply_rotation(vertices, rotation_matrix)
    light_pos = np.array(args.light, dtype=np.float64)

    lighting = Lighting(vertices, faces)
    light_vals = lighting.compute_light_heatmap(light_pos)
    visualize(vertices, faces, light_vals, light_pos)


if __name__ == "__main__":
    main()
