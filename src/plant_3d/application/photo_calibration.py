from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhotoCalibrationEstimate:
    pot_height_phys: float
    unit: str
    pot_height_px: float
    plant_height_px: float
    plant_height_phys: float
    pixels_per_phys: float
    relative_sigma_beta: float


def estimate_plant_height_from_photo(
    *,
    pot_height_px: float,
    plant_height_px: float,
    pot_height_phys: float = 12.5,
    unit: str = "cm",
    pixel_sigma: float = 2.0,
) -> PhotoCalibrationEstimate:
    if pot_height_px <= 0 or plant_height_px <= 0:
        raise ValueError("Pixel heights must be positive.")
    if pot_height_phys <= 0:
        raise ValueError("Reference pot height must be positive.")
    alpha = pot_height_phys / pot_height_px
    plant_height_phys = alpha * plant_height_px
    rel_sigma = math.sqrt((pixel_sigma / plant_height_px) ** 2 + (pixel_sigma / pot_height_px) ** 2)
    return PhotoCalibrationEstimate(
        pot_height_phys=pot_height_phys,
        unit=unit,
        pot_height_px=pot_height_px,
        plant_height_px=plant_height_px,
        plant_height_phys=plant_height_phys,
        pixels_per_phys=pot_height_px / pot_height_phys,
        relative_sigma_beta=rel_sigma,
    )


def mesh_scale_from_photo(*, plant_height_phys: float, mesh_extent_norm: float) -> float:
    if mesh_extent_norm <= 0:
        raise ValueError("Mesh extent along axis must be positive.")
    return plant_height_phys / mesh_extent_norm
