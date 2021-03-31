#!/usr/bin/env python3
"""initialize random centroids """
import numpy as np


def initialize(X, k):
    """returns k centroids from the initial points"""
    if (not isinstance(X, np.ndarray) or not isinstance(k, int) or k <= 0
            or len(X.shape) != 2):
        return None
    maxi = np.max(X, axis=0)
    mini = np.min(X, axis=0)
    res = np.random.uniform(low=mini, high=maxi, size=(k, X.shape[1]))
    return res
