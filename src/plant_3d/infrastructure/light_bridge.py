from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plant_3d.infrastructure.light.core.bvh import build_bvh
from plant_3d.infrastructure.light.geometry.loader import load_glb_to_triangles
from plant_3d.infrastructure.light.geometry.transforms import scale_to_height
from plant_3d.infrastructure.light.io.export_glb import save_colored_glb
from plant_3d.infrastructure.light.io.hdf5 import save_plant_hdf5
from plant_3d.infrastructure.light.lighting.directional import DirectionalLight
from plant_3d.infrastructure.light.lighting.point import PointLight
from plant_3d.infrastructure.light.lighting.simulator import LightSimulator


@dataclass
class LightSimResult:
    output_glb: Path
    output_hdf5: Path
    mean_ppfd: float
    dli: float


def compute_dli(ppfd: np.ndarray, photoperiod: float) -> float:
    seconds_per_day = photoperiod * 3600.0
    return float(np.mean(ppfd) * seconds_per_day * 1e-6)


def run_light_simulation(
    input_glb: str | Path,
    output_dir: str | Path,
    light_type: str = "point",
    light_pos: tuple[float, float, float] = (0.0, 0.0, 0.6),
    light_dir: tuple[float, float, float] = (0.0, 0.0, -1.0),
    power: float = 20.0,
    scale_to_m: float = 0.1,
    photoperiod: float = 12.0,
    add_lamp: bool = False,
    tone_map_mode: str = "relative",
) -> LightSimResult:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    vertices, faces, _ = load_glb_to_triangles(str(input_glb))
    vertices = scale_to_height(vertices, scale_to_m)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    tri_centers = (v0 + v1 + v2) / 3.0
    tri_min = np.minimum(np.minimum(v0, v1), v2)
    tri_max = np.maximum(np.maximum(v0, v1), v2)
    bvh = build_bvh(tri_centers, tri_min, tri_max, range(len(faces)))

    light_pos_arr = np.array(light_pos, dtype=np.float64)
    if light_type == "point":
        light = PointLight(light_pos_arr, power=power)
        light_pos_for_export: np.ndarray | None = light_pos_arr
    else:
        light = DirectionalLight(np.array(light_dir, dtype=np.float64), intensity=power)
        light_pos_for_export = None

    ppfd = LightSimulator(vertices, faces, bvh, v0, v1, v2).compute_vertex_heatmap(light)
    dli = compute_dli(ppfd, photoperiod)
    output_glb = output / "light_output.glb"
    output_hdf5 = output / "light_output.h5"
    save_colored_glb(
        vertices,
        faces,
        ppfd,
        str(output_glb),
        light_pos=light_pos_for_export,
        add_lamp=add_lamp,
        mode=tone_map_mode,
    )
    save_plant_hdf5(
        str(output_hdf5),
        vertices,
        faces,
        ppfd,
        metadata={
            "light_type": light_type,
            "light_pos": list(light_pos),
            "light_dir": list(light_dir),
            "power": power,
            "photoperiod": photoperiod,
            "dli": dli,
            "tone_map_mode": tone_map_mode,
        },
    )
    return LightSimResult(
        output_glb=output_glb,
        output_hdf5=output_hdf5,
        mean_ppfd=float(np.mean(ppfd)),
        dli=dli,
    )
