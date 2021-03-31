#!/usr/bin/env python3
"""optimum number of clusters"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """tests for the optimum number of clusters by variance"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
        return None, None

    if type(kmax) != int or kmax <= 0 or kmax > X.shape[0]:
        return None, None

    if kmin >= kmax:
        return None, None

    if type(iterations) != int or iterations <= 0:
        return None, None
    results = []
    d_vars = []
    centroid, clss = kmeans(X, kmin, iterations)
    results.append(tuple((centroid, clss)))

    d_vars.append(variance(X, centroid) - variance(X, centroid))
    v1 = variance(X, centroid)
    for i in range(kmin+1, kmax+1):
        centroid, clss = kmeans(X, i, iterations)
        results.append(tuple((centroid, clss)))
        d_vars.append(v1 - variance(X, centroid))
    return(results, d_vars)
