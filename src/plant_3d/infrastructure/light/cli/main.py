from __future__ import annotations

import argparse
import numpy as np

from plant_3d.infrastructure.light.geometry.loader import load_glb_to_triangles
from plant_3d.infrastructure.light.geometry.transforms import scale_to_height
from plant_3d.infrastructure.light.core.bvh import build_bvh
from plant_3d.infrastructure.light.lighting.point import PointLight
from plant_3d.infrastructure.light.lighting.directional import DirectionalLight
from plant_3d.infrastructure.light.lighting.simulator import LightSimulator
from plant_3d.infrastructure.light.io.export_glb import save_colored_glb
from plant_3d.infrastructure.light.io.hdf5 import save_plant_hdf5


def compute_dli(ppfd: np.ndarray, photoperiod: float) -> float:
    """
    Compute Daily Light Integral
    """
    seconds_per_day = photoperiod * 3600
    return float(np.mean(ppfd) * seconds_per_day * 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plant light simulation with PPFD and DLI"
    )

    parser.add_argument("input", help="Path to GLB file")
    parser.add_argument(
        "--output", default="./outputs/output.glb", help="Output GLB file"
    )
    parser.add_argument(
        "--hdf5", default="./outputs/result.h5", help="Output HDF5 file"
    )

    parser.add_argument(
        "--light_type", choices=["point", "directional"], default="point"
    )
    parser.add_argument(
        "--light_pos", nargs=3, type=float, default=[0.0, 0.0, 0.6], help="Point light position"
    )
    parser.add_argument(
        "--light_dir", nargs=3, type=float, default=[0.0, 0.0, -1.0], help="Directional light direction"
    )
    parser.add_argument(
        "--power", type=float, default=20.0, help="Light power (µmol/s for point, µmol/m²/s for directional)"
    )

    parser.add_argument(
        "--scale_to_m", type=float, default=0.1, help="Scale model height to meters"
    )

    parser.add_argument(
        "--photoperiod", type=float, default=12.0, help="Hours of light per day (for DLI)"
    )

    parser.add_argument(
        "--add_lamp", action="store_true", help="Add visible lamp to exported GLB"
    )

    args = parser.parse_args()

    vertices, faces, _ = load_glb_to_triangles(args.input)
    vertices = scale_to_height(vertices, args.scale_to_m)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    tri_centers = (v0 + v1 + v2) / 3.0
    tri_min = np.minimum(np.minimum(v0, v1), v2)
    tri_max = np.maximum(np.maximum(v0, v1), v2)
    bvh = build_bvh(tri_centers, tri_min, tri_max, range(len(faces)))

    if args.light_type == "point":
        light_pos = np.array(args.light_pos, dtype=np.float64)
        light = PointLight(light_pos, power=args.power)
    else:
        light_pos = None
        light = DirectionalLight(np.array(args.light_dir, dtype=np.float64), intensity=args.power)

    print(f"Light type: {args.light_type}")
    if light_pos is not None:
        print(f"Light position: {light_pos}")
    print(f"Power: {args.power}")

    simulator = LightSimulator(vertices, faces, bvh, v0, v1, v2)
    ppfd = simulator.compute_vertex_heatmap(light)

    dli = compute_dli(ppfd, args.photoperiod)
    print(f"DLI: {dli:.2f} mol/m²/day")

    save_colored_glb(
        vertices,
        faces,
        ppfd,
        args.output,
        light_pos=light_pos,
        add_lamp=args.add_lamp,
    )

    save_plant_hdf5(
        args.hdf5,
        vertices,
        faces,
        ppfd,
        metadata={
            "light_type": args.light_type,
            "light_pos": args.light_pos if light_pos is not None else [],
            "light_dir": args.light_dir if light_pos is None else [],
            "power": args.power,
            "photoperiod": args.photoperiod,
            "dli": dli,
        },
    )

    print("Done.")


if __name__ == "__main__":
    main()