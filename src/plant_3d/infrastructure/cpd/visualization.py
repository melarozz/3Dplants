'''Interactive 3D visualisation with PyVista.'''
from __future__ import annotations
import logging
import numpy as np
import pyvista as pv
logger = logging.getLogger(__name__)

def _make_pv_cloud(points, scalars = None, scalar_name = 'values'):
    '''Wrap a numpy array as a PyVista PolyData point cloud.'''
    cloud = pv.PolyData(points.astype(np.float32))
    if scalars:
        cloud[scalar_name] = scalars.astype(np.float32)
    return cloud


def show_registration(source, target, displacement_field = None, *, point_size = 3, arrow_scale = 1, arrow_every = 10, window_size = (1600, 900)):
    '''Show interactive 3D registration result.
    '''
    logger.info('Opening PyVista visualisation…')
    plotter = displacement_field(shape = (1, 2) if displacement_field else (1, 1), window_size = window_size, title = 'plant-cpd Registration Result')
    plotter.subplot(0, 0)
    plotter.add_text('Aligned overlay', font_size = 12, color = 'white')
    src_cloud = _make_pv_cloud(source)
    plotter.add_points(src_cloud, color = '#50C878', point_size = point_size, label = 'Source (aligned)', render_points_as_spheres = True)
    tgt_cloud = _make_pv_cloud(target)
    plotter.add_points(tgt_cloud, color = '#6495ED', point_size = point_size, label = 'Target', render_points_as_spheres = True)
    plotter.add_legend(bcolor = 'black', face = None)
    if displacement_field:
        plotter.add_text('Displacement field', font_size = 12, color = 'white')
        magnitudes = np.linalg.norm(displacement_field, axis = 1)
        idx = np.arange(0, len(source), arrow_every)
        pts_sub = source[idx]
        disp_sub = displacement_field[idx]
        mag_sub = magnitudes[idx]
        cloud_disp = _make_pv_cloud(source, magnitudes, 'displacement')
        plotter.add_points(cloud_disp, scalars = 'displacement', cmap = 'plasma', point_size = point_size, render_points_as_spheres = True, scalar_bar_args = {
            'title': '‖d‖' })
        arrow_cloud = pv.PolyData(pts_sub.astype(np.float32))
        arrow_cloud['vectors'] = (disp_sub * arrow_scale).astype(np.float32)
        arrow_cloud['mag'] = mag_sub.astype(np.float32)
        arrows = arrow_cloud.glyph(orient = 'vectors', scale = 'mag', factor = arrow_scale)
        plotter.add_mesh(arrows, cmap = 'plasma', opacity = 0.8)
    plotter.link_views()
    plotter.show()

