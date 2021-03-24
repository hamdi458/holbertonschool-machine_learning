#!/usr/bin/env python3
import numpy as np


def pca(X, ndim):
    mean = np.mean(X, axis=0)

    go = X-mean
    u, s, vh = np.linalg.svd(go)
    Ureduce = vh[: ndim]
    t = np.matmul(go, Ureduce.T)

    return t
