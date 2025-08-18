import numpy as np
import pyvista as pv


def visualize(vertices: np.ndarray,
              faces: np.ndarray,
              light_values: np.ndarray,
              light_pos: np.ndarray,
              add_lamp: bool = True):
    """
    Visualize mesh with per-vertex light values and an optional small sphere for the light position.

    Parameters
    ----------
    vertices : (N,3) array
    faces : (M,3) int array
    light_values : (N,) array of scalars (0..1)
    light_pos : (3,) array world coordinates of lamp
    add_lamp : bool
        If True, add a small yellow sphere at light_pos in the scene.
    """
    # pyvista expects faces in the "n, i, j, k" flattened format
    faces_pv = np.hstack([[3, *f] for f in faces])
    pv_mesh = pv.PolyData(vertices, faces_pv)
    pv_mesh.point_data["light"] = light_values

    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="light", cmap="plasma", smooth_shading=True)

    if add_lamp:
        # sphere radius as ~1% of model bbox diagonal, min radius to be visible
        bbox_diag = np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0))
        sphere_radius = max(0.01, bbox_diag * 0.01)
        lamp_sphere = pv.Sphere(radius=sphere_radius, center=light_pos)
        plotter.add_mesh(lamp_sphere, color="yellow", smooth_shading=True)

    plotter.show_axes()
    plotter.show()
