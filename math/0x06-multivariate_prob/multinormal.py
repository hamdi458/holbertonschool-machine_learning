#!/usr/bin/env python3
"""Class MultiNormal"""
import numpy as np


class MultiNormal(object):
    """Class MultiNormal"""

    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[0] < 2:
            raise ValueError("data must contain multiple data points")
        n = data.shape[1] - 1
        self.mean = np.mean(data, axis=1, keepdims=True)
        X = data
        self.cov = 1/n * np.dot((X - self.mean), (X - self.mean).T)
