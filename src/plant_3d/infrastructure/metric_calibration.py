"""Uniform metric scaling for meshes/point clouds"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _axis_index(axis: str) -> int:
    key = axis.lower().strip()
    if key not in _AXIS_INDEX:
        raise ValueError(f"axis must be one of {list(_AXIS_INDEX)}, got {axis!r}")
    return _AXIS_INDEX[key]


def unit_to_meters(value: float, unit: str) -> float:
    u = unit.lower().strip()
    if u in ("m", "meter", "meters"):
        return float(value)
    if u in ("cm", "centimeter", "centimeters"):
        return float(value) * 0.01
    if u in ("mm", "millimeter", "millimeters"):
        return float(value) * 0.001
    raise ValueError(f"Unsupported unit {unit!r}; use m, cm, or mm")


def load_vertices(path: str | Path) -> np.ndarray:
    p = Path(path).expanduser().resolve()
    loaded = trimesh.load(str(p))
    if isinstance(loaded, trimesh.Scene):
        geom = loaded.dump(concatenate=True)
    else:
        geom = loaded
    if hasattr(geom, "vertices"):
        return np.asarray(geom.vertices, dtype=np.float64)
    raise TypeError(f"Unsupported geometry type from {p}: {type(geom)}")


def extent_along_axis(vertices: np.ndarray, axis: int) -> float:
    if vertices.size == 0:
        return 0.0
    lo = float(np.min(vertices[:, axis]))
    hi = float(np.max(vertices[:, axis]))
    return hi - lo


def shift_min_axis_to_zero(vertices: np.ndarray, axis: int) -> np.ndarray:
    out = vertices.copy()
    out[:, axis] -= float(np.min(out[:, axis]))
    return out


@dataclass(frozen=True)
class CalibrationResult:
    mode: str
    scale_factor: float | None
    axis: str
    extent_reference: float | None
    target_extent_m: float | None
    reference_path: str | None


def compute_scale_from_reference_extent(
    reference_vertices: np.ndarray,
    *,
    target_extent_m: float,
    axis: str,
) -> tuple[float, float]:
    """
    K = target_extent_m / extent_ref so that reference mesh extent along axis becomes target.
    Returns (scale_factor, extent_before).
    """
    ax = _axis_index(axis)
    ext = extent_along_axis(reference_vertices, ax)
    if ext <= 0:
        raise ValueError("Reference extent along axis is zero; cannot calibrate.")
    return target_extent_m / ext, ext


def apply_uniform_scale(
    vertices: np.ndarray,
    factor: float,
    *,
    shift_min_axis: int | None,
) -> np.ndarray:
    out = np.asarray(vertices, dtype=np.float64) * float(factor)
    if shift_min_axis is not None:
        out = shift_min_axis_to_zero(out, shift_min_axis)
    return out


def save_mesh_like_input(
    input_path: str | Path,
    vertices: np.ndarray,
    output_path: str | Path,
) -> Path:
    inp = Path(input_path).expanduser().resolve()
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    loaded = trimesh.load(str(inp))
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if not hasattr(mesh, "vertices"):
        raise TypeError(f"Expected mesh-like geometry from {inp}")
    v_old = np.asarray(mesh.vertices, dtype=np.float64)
    if v_old.shape != vertices.shape:
        raise ValueError(
            f"Vertex count mismatch for {inp}: mesh has {v_old.shape[0]}, scaled has {vertices.shape[0]}",
        )
    mesh.vertices = vertices
    mesh.export(str(out))
    return out


def calibrate_files(
    input_paths: list[Path],
    output_dir: Path,
    *,
    reference_path: Path | None,
    physical_extent: float | None,
    unit: str,
    axis: str,
    uniform_scale: float | None,
    shift_min_to_zero: bool,
    name_suffix: str,
) -> tuple[list[Path], CalibrationResult]:
    """
    Scale meshes uniformly.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ax = _axis_index(axis)
    shift_axis: int | None = ax if shift_min_to_zero else None

    if uniform_scale is not None and physical_extent is not None:
        raise ValueError("Use either --uniform-scale or --physical-extent, not both.")
    if uniform_scale is None and physical_extent is None:
        raise ValueError("Provide --physical-extent or --uniform-scale.")

    written: list[Path] = []
    result_summary: CalibrationResult

    if uniform_scale is not None:
        k = float(uniform_scale)
        for inp in input_paths:
            inp = inp.expanduser().resolve()
            verts = load_vertices(inp)
            scaled = apply_uniform_scale(verts, k, shift_min_axis=shift_axis)
            stem = inp.stem + f"_{name_suffix}"
            out_path = output_dir / f"{stem}{inp.suffix.lower()}"
            save_mesh_like_input(inp, scaled, out_path)
            written.append(out_path)
        result_summary = CalibrationResult(
            mode="uniform_scale",
            scale_factor=k,
            axis=axis,
            extent_reference=None,
            target_extent_m=None,
            reference_path=None,
        )
        return written, result_summary

    assert physical_extent is not None
    target_m = unit_to_meters(physical_extent, unit)

    if reference_path is not None:
        ref_p = reference_path.expanduser().resolve()
        ref_v = load_vertices(ref_p)
        k, ext_ref = compute_scale_from_reference_extent(ref_v, target_extent_m=target_m, axis=axis)
        result_summary = CalibrationResult(
            mode="reference_extent",
            scale_factor=k,
            axis=axis,
            extent_reference=ext_ref,
            target_extent_m=target_m,
            reference_path=str(ref_p),
        )
        for inp in input_paths:
            inp = inp.expanduser().resolve()
            verts = load_vertices(inp)
            scaled = apply_uniform_scale(verts, k, shift_min_axis=shift_axis)
            stem = inp.stem + f"_{name_suffix}"
            out_path = output_dir / f"{stem}{inp.suffix.lower()}"
            save_mesh_like_input(inp, scaled, out_path)
            written.append(out_path)
        return written, result_summary

    result_summary = CalibrationResult(
        mode="per_file_extent",
        scale_factor=None,
        axis=axis,
        extent_reference=None,
        target_extent_m=target_m,
        reference_path=None,
    )
    for inp in input_paths:
        inp = inp.expanduser().resolve()
        verts = load_vertices(inp)
        k, _ext = compute_scale_from_reference_extent(verts, target_extent_m=target_m, axis=axis)
        scaled = apply_uniform_scale(verts, k, shift_min_axis=shift_axis)
        stem = inp.stem + f"_{name_suffix}"
        out_path = output_dir / f"{stem}{inp.suffix.lower()}"
        save_mesh_like_input(inp, scaled, out_path)
        written.append(out_path)
    return written, result_summary


def calibrate_from_photo_measurements(
    input_paths: list[str | Path],
    output_dir: str | Path,
    *,
    pot_height_px: float,
    plant_height_px: float,
    pot_height_phys: float = 12.5,
    unit: str = "cm",
    pixel_sigma: float = 2.0,
    axis: str = "z",
    shift_min_to_zero: bool = False,
    name_suffix: str = "metric",
):
    from plant_3d.application.photo_calibration import (
        PhotoCalibrationEstimate,
        estimate_plant_height_from_photo,
        mesh_scale_from_photo,
    )

    photo = estimate_plant_height_from_photo(
        pot_height_px=pot_height_px,
        plant_height_px=plant_height_px,
        pot_height_phys=pot_height_phys,
        unit=unit,
        pixel_sigma=pixel_sigma,
    )
    paths = [Path(x).expanduser().resolve() for x in input_paths]
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ax = _axis_index(axis)
    shift_axis = ax if shift_min_to_zero else None
    written: list[Path] = []
    for inp in paths:
        verts = load_vertices(inp)
        ext = extent_along_axis(verts, ax)
        k = mesh_scale_from_photo(plant_height_phys=photo.plant_height_phys, mesh_extent_norm=ext)
        scaled = apply_uniform_scale(verts, k, shift_min_axis=shift_axis)
        dest = out / f"{inp.stem}_{name_suffix}{inp.suffix.lower()}"
        save_mesh_like_input(inp, scaled, dest)
        written.append(dest)
    summary = CalibrationResult(
        mode="photo_extent",
        scale_factor=k,
        axis=axis,
        extent_reference=ext,
        target_extent_m=unit_to_meters(photo.plant_height_phys, unit),
        reference_path=None,
    )
    return written, summary, photo
