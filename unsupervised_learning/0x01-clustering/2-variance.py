#!/usr/bin/env python3
"""variance kmeans"""
import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance for a data set X"""
    if (not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray)
            or len(X.shape) != 2 or len(C.shape) != 2
            or X.shape[1] != C.shape[1]):
        return None
    distance = (((X - C[:, np.newaxis])**2).sum(axis=2))
    if C.shape[0] > 1:
        min_distance = np.min(distance, axis=0)
        v = np.sum(min_distance)
    else:
        v = np.sum(distance)
    return v
