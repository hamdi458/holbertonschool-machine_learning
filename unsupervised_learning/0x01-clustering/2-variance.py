#!/usr/bin/env python3
"""variance kmeans"""
import numpy as np


def variance(X, C):
    """alculates the total intra-cluster variance for a data set X"""
    distance = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
    min_distance = np.min(distance, axis=0)
    v = np.sum(min_distance ** 2)
    return v
