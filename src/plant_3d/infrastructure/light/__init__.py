"""Ray-tracing light simulation on 3D plant meshes."""

from importlib.metadata import PackageNotFoundError, version

from plant_3d.infrastructure.light.geometry.loader import load_glb_to_triangles
from plant_3d.infrastructure.light.geometry.transforms import apply_rotation
from plant_3d.infrastructure.light.io.export_glb import save_colored_glb
from plant_3d.infrastructure.light.lighting.directional import DirectionalLight
from plant_3d.infrastructure.light.lighting.point import PointLight
from plant_3d.infrastructure.light.lighting.simulator import LightSimulator

try:
    __version__ = version("plant-cpd")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "load_glb_to_triangles",
    "apply_rotation",
    "LightSimulator",
    "PointLight",
    "DirectionalLight",
    "save_colored_glb",
]
