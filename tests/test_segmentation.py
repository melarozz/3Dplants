from __future__ import annotations

import numpy as np

from plant_3d.infrastructure.segmentation import LeafStemSegmentationConfig, segment_leaf_stem


def test_segment_leaf_stem_synthetic() -> None:
    v = np.array([[0, 0, 0], [0.01, 0, 0.5], [0.2, 0, 0.5], [0.21, 0, 0.51]], dtype=np.float64)
    f = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    out = segment_leaf_stem(v, f, LeafStemSegmentationConfig(min_component_vertices=1))
    assert out["labels_vertex"] is not None
