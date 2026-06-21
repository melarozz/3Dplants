from __future__ import annotations
from pathlib import Path
from plant_3d.application.models import PostSegResult, PrepSegResult
from plant_3d.application.types import LabeledPointCloud
from plant_3d.infrastructure.io.formats import labels_to_colors, load_pcd_ascii, load_stl_points, parse_labels_txt, save_label_metadata, save_labeled_ply, save_pcd_ascii, voxel_downsample

def run_prep_seg(input_stl, output_dir, voxel_size):
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents = True, exist_ok = True)
    points = load_stl_points(input_stl)
    sampled = voxel_downsample(points, voxel_size = voxel_size)
    pcd_path = save_pcd_ascii(output / 'prepared_cloud.pcd', sampled)
    return PrepSegResult(pcd_path = pcd_path, points_count = sampled.shape[0])


def run_post_seg(points_source, labels_txt, output_dir, class_map):
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents = True, exist_ok = True)
    source = Path(points_source).expanduser().resolve()
    if source.suffix.lower() == '.pcd':
        points = load_pcd_ascii(source)
    else:
        points = load_stl_points(source)
    labels = parse_labels_txt(labels_txt)
    if labels.shape[0] != points.shape[0]:
        msg = f'''TXT labels size {labels.shape[0]} != point count {points.shape[0]}'''
        raise ValueError(msg)
    cloud = LabeledPointCloud(points = points, labels = labels, colors = labels_to_colors(labels, class_map))
    ply_path = save_labeled_ply(output / 'labeled.ply', cloud)
    labels_json = save_label_metadata(output / 'labels.json', labels, class_map)
    class_map_json = save_label_metadata(output / 'class_map.json', labels, class_map)
    return PostSegResult(ply_path = ply_path, labels_json = labels_json, class_map_json = class_map_json)

