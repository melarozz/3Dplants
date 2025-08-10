import numpy as np
import time
from typing import Optional, Sequence
from ..intersection import ray_triangle_intersect
from ..ray import Ray


class AABB:
    def __init__(self, min_pt, max_pt):
        self.min = np.array(min_pt, dtype=np.float64)
        self.max = np.array(max_pt, dtype=np.float64)

    def intersect(self, ray: Ray, t_min_allowed: float = 0.0) -> bool:
        """
        Slab method intersection test. Uses inverse direction for speed.
        Returns True if ray intersects AABB at t >= t_min_allowed.
        """
        # avoid division by zero
        dir_inv = 1.0 / (ray.d + 1e-20)
        tmin = (self.min - ray.o) * dir_inv
        tmax = (self.max - ray.o) * dir_inv
        t1 = np.minimum(tmin, tmax)
        t2 = np.maximum(tmin, tmax)
        t_enter = np.max(t1)
        t_exit = np.min(t2)
        return (t_exit >= max(t_enter, t_min_allowed))


class BVHNode:
    def __init__(self, indices: Optional[np.ndarray], aabb: AABB, left=None, right=None):
        # if indices is not None -> leaf
        self.indices = indices
        self.aabb = aabb
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.indices is not None


def build_bvh(tri_centers: np.ndarray, tri_min: np.ndarray, tri_max: np.ndarray,
              tri_indices: Sequence[int], max_leaf: int = 8) -> BVHNode:
    """
    Build a binary BVH recursively. tri_centers, tri_min, tri_max are arrays indexed by triangle index.
    """

    def recurse(indices: np.ndarray) -> BVHNode:
        node_aabb = AABB(np.min(tri_min[indices], axis=0),
                         np.max(tri_max[indices], axis=0))
        if len(indices) <= max_leaf:
            return BVHNode(indices, node_aabb)
        span = node_aabb.max - node_aabb.min
        axis = int(np.argmax(span))
        order = indices[np.argsort(tri_centers[indices, axis])]
        mid = len(order) // 2
        left = recurse(order[:mid])
        right = recurse(order[mid:])
        return BVHNode(None, node_aabb, left, right)

    print(f"[BVH] Building with {len(tri_indices)} triangles...")
    start = time.time()
    root = recurse(np.array(tri_indices, dtype=np.int64))
    print(f"[BVH] Built in {time.time() - start:.2f}s")
    return root


def traverse_bvh(root: BVHNode, ray: Ray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray):
    """
    Traverse BVH with an explicit stack. Returns hit_info (t, triangle_index, u, v) or None.
    """
    hit_t = np.inf
    hit_info = None
    stack = [root]

    while stack:
        node = stack.pop()
        if node is None:
            continue
        if not node.aabb.intersect(ray):
            continue
        if node.is_leaf:
            # iterate triangles in leaf
            for idx in node.indices:
                res = ray_triangle_intersect(ray.o, ray.d, v0[idx], v1[idx], v2[idx])
                if res:
                    t, u, v = res
                    if t < hit_t:
                        hit_t = t
                        hit_info = (t, int(idx), u, v)
        else:
            # push children; order doesn't attempt near-first optimization
            stack.append(node.left)
            stack.append(node.right)

    return hit_info
