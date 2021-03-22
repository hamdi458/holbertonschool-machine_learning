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
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        n = data.shape[1] - 1
        self.mean = np.mean(data, axis=1, keepdims=True)
        X = data
        self.cov = 1/n * np.dot((X - self.mean), (X - self.mean).T)

    def pdf(self, x):
        """ value pdf"""

        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')

        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        if x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError('x must have the shape ({}, 1)'.format(d))
        n, _ = x.shape
        var1 = np.sqrt(((2 * np.pi) ** n) * np.linalg.det(self.cov))
        var3 = np.dot((x - self.mean).T, np.linalg.inv(self.cov))
        var2 = (-0.5 * np.dot(var3, x - self.mean))
        pdf = (1 / var1) * np.exp(var2[0][0])
        return pdf
