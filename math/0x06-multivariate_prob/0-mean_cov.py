#!/usr/bin/env python3
"""calculates the mean and covariance of a data set"""
import numpy as np


def mean_cov(X):
    """calculates the mean and covariance of a data set"""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")
    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    n = X.shape[0] - 1
    mean = np.mean(X, axis=0, keepdims=True)
    return mean, 1/n * np.dot((X - mean).T, (X - mean))
