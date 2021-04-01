#!/usr/bin/env python3
"""Gaussian Mixture Model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """initializes variables for a Gaussian Mixture Model"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k < 1:
        return None, None, None
    d = X.shape[1]
    pi = np.ones((k)) / k
    m, clss = kmeans(X, k)
    S = np.tile(np.identity(d), (k, 1, 1))
    return pi, m, S
