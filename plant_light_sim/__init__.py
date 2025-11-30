"""
plant_light_sim

Library for simulating light distribution on 3D plant meshes.
"""

from importlib.metadata import version, PackageNotFoundError

# --- Version ---
try:
    __version__ = version("plant-light-sim")
except PackageNotFoundError:
    __version__ = "0.0.0"


# --- Core API (high-level imports) ---
from plant_light_sim.geometry.loader import load_glb_to_triangles
from plant_light_sim.geometry.transforms import apply_rotation

from plant_light_sim.lighting.simulator import LightSimulator
from plant_light_sim.lighting.point import PointLight
from plant_light_sim.lighting.directional import DirectionalLight

from plant_light_sim.io.export_glb import save_colored_glb


# --- Public interface ---
__all__ = [
    "__version__",
    # geometry
    "load_glb_to_triangles",
    "apply_rotation",
    # lighting
    "LightSimulator",
    "PointLight",
    "DirectionalLight",
    # io
    "save_colored_glb",
]