#!/usr/bin/env python3
""" pca """
import numpy as np


def pca(X, var=0.95):
    """ Function that performs PCA on a dataset: """
    s = np.linalg.svd(X)[1]

    variance = np.cumsum(s) / np.sum(s)
    d = np.argwhere(variance >= var)[0, 0]
    vh = np.linalg.svd(X)[2]
    return vh[:d + 1].T
