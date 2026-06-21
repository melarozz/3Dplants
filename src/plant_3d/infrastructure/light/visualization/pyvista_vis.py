from __future__ import annotations

import numpy as np
import pyvista as pv


def visualize(
    vertices: np.ndarray,
    faces: np.ndarray,
    heatmap: np.ndarray,
) -> None:
    """
    Interactive visualization using PyVista.
    """
    faces_pv = np.hstack([[3, *f] for f in faces])

    mesh = pv.PolyData(vertices, faces_pv)
    mesh.point_data["light"] = heatmap

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="light", cmap="plasma", smooth_shading=True)
    plotter.show_axes()
    plotter.show()