from __future__ import annotations

import click
import numpy as np
import trimesh

from plant_3d.infrastructure.segmentation import LeafStemSegmentationConfig, segment_leaf_stem


@click.command()
@click.argument("input_mesh", type=click.Path(exists=True))
@click.option("--output", "-o", default="./outputs/segmentation_labels.npy", show_default=True)
def main(input_mesh: str, output: str) -> None:
    mesh = trimesh.load(input_mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    out = segment_leaf_stem(vertices, faces, LeafStemSegmentationConfig())
    np.save(output, out["labels_vertex"])
    click.echo(f"Saved labels: {output}")


if __name__ == "__main__":
    main()

