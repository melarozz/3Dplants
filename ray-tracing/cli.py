import argparse
import os
import numpy as np
import trimesh
from .geometry import load_glb_to_triangles, apply_rotation
from .lighting import Lighting
from .visualization import visualize


def save_colored_glb(vertices: np.ndarray, faces: np.ndarray, light_vals: np.ndarray,
                     out_filename: str, light_pos: np.ndarray, add_lamp: bool = True):
    """
    Create a mesh with per-vertex colors from light_vals, optionally add a small sphere at light_pos,
    and save as GLB. Uses matplotlib 'plasma' colormap to convert scalars -> rgba bytes.
    """
    try:
        import matplotlib.cm as cm
    except Exception as e:
        raise RuntimeError("matplotlib is required to map heatmap to colors for GLB export.") from e

    # clamp values to [0,1]
    lv = np.clip(light_vals, 0.0, 1.0)
    cmap = cm.get_cmap("plasma")
    rgba = cmap(lv)  # Nx4 floats 0..1
    colors = (rgba * 255).astype(np.uint8)  # Nx4 uint8

    export_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    export_mesh.visual.vertex_colors = colors

    meshes_to_concat = [export_mesh]

    if add_lamp:
        # create lamp sphere and place it at light_pos
        # sphere radius = max(0.01m, 1% of bbox diagonal)
        bbox_diag = np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0))
        sphere_radius = max(0.01, bbox_diag * 0.01)

        # create an icosphere (subdivisions=3 gives reasonable detail), translate to light_pos
        sphere_mesh = trimesh.creation.icosphere(subdivisions=3, radius=sphere_radius)
        sphere_mesh.apply_translation(light_pos)

        # color the sphere yellow (RGBA uint8)
        yellow = np.array([255, 225, 0, 255], dtype=np.uint8)  # slightly warm yellow
        sphere_colors = np.tile(yellow, (len(sphere_mesh.vertices), 1))
        sphere_mesh.visual.vertex_colors = sphere_colors

        meshes_to_concat.append(sphere_mesh)

    # concatenate meshes (preserves per-vertex colors)
    if len(meshes_to_concat) > 1:
        combined = trimesh.util.concatenate(meshes_to_concat)
    else:
        combined = meshes_to_concat[0]

    print(f"[Export] Saving colored GLB to: {out_filename}")
    combined.export(out_filename)
    print("[Export] Finished.")


def main():
    parser = argparse.ArgumentParser(description="GLB Light Heatmap Visualizer (scaled & exported)")
    parser.add_argument("glb_file", help="Path to GLB/GLTF/GLB file")
    parser.add_argument("--light", nargs=3, type=float, default=[0.0, 0.0, 1.0],
                        help="Light position (x y z). z will be overridden to 0.60 m (60 cm).")
    parser.add_argument("--rotation", nargs=9, type=float,
                        help="Optional 3x3 rotation matrix in row-major order (9 numbers)")
    parser.add_argument("--scale_to_m", type=float, default=0.10,
                        help="Scale model so its Z-height becomes this value (meters). Default 0.10 (10 cm).")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Do not open interactive PyVista visualization (just export GLB).")
    # lamp toggle: default is to include lamp; pass --no-lamp to disable
    parser.add_argument("--no-lamp", dest="add_lamp", action="store_false",
                        help="Do not include lamp sphere in exported GLB or visualization.")
    parser.set_defaults(add_lamp=True)

    args = parser.parse_args()

    vertices, faces, mesh = load_glb_to_triangles(args.glb_file)

    # rotation
    if args.rotation:
        rotation_matrix = np.array(args.rotation, dtype=np.float64).reshape((3, 3))
        vertices = apply_rotation(vertices, rotation_matrix)
    else:
        # default rotation (same as before)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=np.float64)
        vertices = apply_rotation(vertices, rotation_matrix)

    # --- Scale model so its Z-height (maxZ - minZ) becomes args.scale_to_m (default 0.10 m)
    z_min = np.min(vertices[:, 2])
    z_max = np.max(vertices[:, 2])
    current_height = z_max - z_min
    desired_height = float(args.scale_to_m) if args.scale_to_m > 0 else 0.10
    if current_height > 0:
        scale = desired_height / current_height
        vertices = vertices * scale
        # after scaling, translate so floor (min z) is at z = 0
        vertices[:, 2] -= np.min(vertices[:, 2])
        print(f"[Scale] Scaled model by {scale:.6f} so height -> {desired_height} m and placed on floor (z>=0).")
    else:
        print("[Scale] Model has zero Z-span, skipping scaling.")

    # Light position: keep x,y if provided, but force z to 0.60 (60 cm from floor)
    light_pos = np.array(args.light, dtype=np.float64)
    light_pos[2] = 0.60
    print(f"[Light] Using light position (z forced to 0.60 m): {light_pos}")

    lighting = Lighting(vertices, faces)
    light_vals = lighting.compute_light_heatmap(light_pos)

    # visualize unless disabled
    if not args.no_visualize:
        try:
            visualize(vertices, faces, light_vals, light_pos, add_lamp=args.add_lamp)
        except Exception as e:
            print(f"[Visualize] Visualization failed: {e}. Continuing to export GLB.")

    # export colored glb next to input filename, optionally include lamp sphere
    base, ext = os.path.splitext(args.glb_file)
    out_filename = f"{base}_lighted.glb"
    save_colored_glb(vertices, faces, light_vals, out_filename, light_pos, add_lamp=args.add_lamp)
    print(f"[Done] Exported colored model to: {out_filename}")


if __name__ == "__main__":
    main()
