#!/usr/bin/env python3
import numpy as np


def initialize(X, k):
    """returns k centroids from the initial points"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    maxi = np.max(X, axis=0)
    mini = np.min(X, axis=0)
    res = np.random.uniform(low=mini, high=maxi, size=(k, X.shape[1]))
    return res
