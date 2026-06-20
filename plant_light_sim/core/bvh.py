from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from plant_light_sim.core.aabb import AABB
from plant_light_sim.core.intersection import ray_triangle_intersect
from plant_light_sim.core.ray import Ray


class BVHNode:
    """
    Node of Bounding Volume Hierarchy.
    """

    def __init__(
        self,
        indices: Optional[np.ndarray],
        aabb: AABB,
        left: Optional["BVHNode"] = None,
        right: Optional["BVHNode"] = None,
    ) -> None:
        self.indices = indices  # leaf if not None
        self.aabb = aabb
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        return self.indices is not None


def build_bvh(
    tri_centers: np.ndarray,
    tri_min: np.ndarray,
    tri_max: np.ndarray,
    tri_indices: Sequence[int],
    max_leaf: int = 8,
) -> BVHNode:
    """
    Build BVH recursively.
    """

    def recurse(indices: np.ndarray) -> BVHNode:
        node_aabb = AABB(
            np.min(tri_min[indices], axis=0),
            np.max(tri_max[indices], axis=0),
        )

        if len(indices) <= max_leaf:
            return BVHNode(indices, node_aabb)

        span = node_aabb.max - node_aabb.min
        axis = int(np.argmax(span))

        order = indices[np.argsort(tri_centers[indices, axis])]
        mid = len(order) // 2

        left = recurse(order[:mid])
        right = recurse(order[mid:])

        return BVHNode(None, node_aabb, left, right)

    start = time.time()
    root = recurse(np.array(tri_indices, dtype=np.int64))
    elapsed = time.time() - start

    print(f"[BVH] Built in {elapsed:.2f}s")

    return root


def traverse_bvh(
    root: BVHNode,
    ray: Ray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> tuple[float, int, float, float] | None:
    """
    Traverse BVH and return closest hit.

    Returns
    -------
    (t, triangle_index, u, v) or None
    """
    hit_t = np.inf
    hit_info: tuple[float, int, float, float] | None = None

    stack: list[BVHNode] = [root]

    while stack:
        node = stack.pop()

        if node is None:
            continue

        if not node.aabb.intersect(ray):
            continue

        if node.is_leaf:
            for idx in node.indices:
                res = ray_triangle_intersect(
                    ray.o, ray.d,
                    v0[idx], v1[idx], v2[idx]
                )

                if res:
                    t, u, v = res
                    if t < hit_t:
                        hit_t = t
                        hit_info = (t, int(idx), u, v)

        else:
            # no near-first ordering (can be improved later)
            stack.append(node.left)
            stack.append(node.right)

    return hit_info