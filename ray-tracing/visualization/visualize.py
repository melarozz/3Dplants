import numpy as np
import pyvista as pv


def visualize(vertices: np.ndarray, faces: np.ndarray, light_values: np.ndarray, light_pos: np.ndarray):
    """
    Visualize mesh with per-vertex light values and a small sphere for the light position.
    """
    faces_pv = np.hstack([[3, *f] for f in faces])
    pv_mesh = pv.PolyData(vertices, faces_pv)
    pv_mesh.point_data["light"] = light_values

    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="light", cmap="plasma", smooth_shading=True)
    lamp_sphere = pv.Sphere(
        radius=max(0.01, np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0)) * 0.01),
        center=light_pos)
    plotter.add_mesh(lamp_sphere, color="yellow", smooth_shading=True)
    plotter.show_axes()
    plotter.show()
