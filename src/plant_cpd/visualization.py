"""Interactive 3D visualisation with PyVista."""

from __future__ import annotations

import logging

import numpy as np
import pyvista as pv

logger = logging.getLogger(__name__)


def _make_pv_cloud(
    points: np.ndarray,
    scalars: np.ndarray | None = None,
    scalar_name: str = "values",
) -> pv.PolyData:
    """Wrap a numpy array as a PyVista PolyData point cloud."""
    cloud = pv.PolyData(points.astype(np.float32))
    if scalars is not None:
        cloud[scalar_name] = scalars.astype(np.float32)
    return cloud


def show_registration(
    source: np.ndarray,
    target: np.ndarray,
    displacement_field: np.ndarray | None = None,
    *,
    point_size: float = 3.0,
    arrow_scale: float = 1.0,
    arrow_every: int = 10,
    window_size: tuple[int, int] = (1600, 900),
) -> None:
    """Show interactive 3D registration result.

    Displays source (green) and target (blue) point clouds.
    If *displacement_field* is provided, also shows
    displacement arrows on a sub-sampled set of points.

    Parameters
    ----------
    source : np.ndarray
        Aligned source points ``(M, 3)``.
    target : np.ndarray
        Target points ``(N, 3)``.
    displacement_field : np.ndarray | None
        Displacement vectors ``(M, 3)`` from non-rigid step.
    point_size : float
        Rendering point size.
    arrow_scale : float
        Scale factor for displacement arrows.
    arrow_every : int
        Show an arrow every *n*-th point (for readability).
    window_size : tuple[int, int]
        Window dimensions ``(width, height)``.
    """
    logger.info("Opening PyVista visualisation…")

    plotter = pv.Plotter(
        shape=(1, 2) if displacement_field is not None else (1, 1),
        window_size=window_size,
        title="plant-cpd Registration Result",
    )

    # ── Left: aligned overlay ─────────────────────────────
    plotter.subplot(0, 0)
    plotter.add_text(
        "Aligned overlay", font_size=12, color="white",
    )

    src_cloud = _make_pv_cloud(source)
    plotter.add_points(
        src_cloud,
        color="#50C878",
        point_size=point_size,
        label="Source (aligned)",
        render_points_as_spheres=True,
    )

    tgt_cloud = _make_pv_cloud(target)
    plotter.add_points(
        tgt_cloud,
        color="#6495ED",
        point_size=point_size,
        label="Target",
        render_points_as_spheres=True,
    )
    plotter.add_legend(bcolor="black", face=None)

    # ── Right: displacement field ─────────────────────────
    if displacement_field is not None:
        plotter.subplot(0, 1)
        plotter.add_text(
            "Displacement field", font_size=12, color="white",
        )

        magnitudes = np.linalg.norm(displacement_field, axis=1)

        # Sub-sample for readability
        idx = np.arange(0, len(source), arrow_every)
        pts_sub = source[idx]
        disp_sub = displacement_field[idx]
        mag_sub = magnitudes[idx]

        # Points coloured by displacement magnitude
        cloud_disp = _make_pv_cloud(
            source, magnitudes, "displacement",
        )
        plotter.add_points(
            cloud_disp,
            scalars="displacement",
            cmap="plasma",
            point_size=point_size,
            render_points_as_spheres=True,
            scalar_bar_args={"title": "‖d‖"},
        )

        # Arrows
        arrow_cloud = pv.PolyData(pts_sub.astype(np.float32))
        arrow_cloud["vectors"] = (
            disp_sub * arrow_scale
        ).astype(np.float32)
        arrow_cloud["mag"] = mag_sub.astype(np.float32)
        arrows = arrow_cloud.glyph(
            orient="vectors",
            scale="mag",
            factor=arrow_scale,
        )
        plotter.add_mesh(
            arrows, cmap="plasma", opacity=0.8,
        )

    plotter.link_views()
    plotter.show()