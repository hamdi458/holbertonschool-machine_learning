#!/usr/bin/env python3
"""algo PCA"""
import numpy as np


def pca(X, ndim):
    """performe algo PCA"""
    mean = np.mean(X, axis=0)

    go = X-mean
    u, s, vh = np.linalg.svd(go)
    Ureduce = vh[: ndim]
    t = np.matmul(go, Ureduce.T)

    return t
