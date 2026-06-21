'''Registration quality metrics.'''
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

def chamfer_distance(a, b):
    '''Symmetric Chamfer distance between two point clouds.
    '''
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    (dists_a2b, _) = tree_b.query(a)
    (dists_b2a, _) = tree_a.query(b)
    return float((dists_a2b ** 2).mean() + (dists_b2a ** 2).mean())


def hausdorff_distance(a, b):
    '''Symmetric Hausdorff distance.
    '''
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    (dists_a2b, _) = tree_b.query(a)
    (dists_b2a, _) = tree_a.query(b)
    return float(max(dists_a2b.max(), dists_b2a.max()))


def rmse(a, b):
    '''Root Mean Square Error of nearest-neighbour distances.
    '''
    tree_b = cKDTree(b)
    (dists, _) = tree_b.query(a)
    return float(np.sqrt((dists ** 2).mean()))


def inlier_ratio(a, b, threshold = 0.05):
    '''Fraction of source points within *threshold* of target.
    '''
    tree_b = cKDTree(b)
    (dists, _) = tree_b.query(a)
    return float((dists < threshold).mean())


def compute_all_metrics(source, target, inlier_threshold = 0.05):
    '''Compute all registration quality metrics.
    '''
    return {
        'chamfer': chamfer_distance(source, target),
        'hausdorff': hausdorff_distance(source, target),
        'rmse': rmse(source, target),
        'inlier_ratio': inlier_ratio(source, target, inlier_threshold) }

