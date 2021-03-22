#!/usr/bin/env python3
""" calculates a correlation matrix"""
import numpy as np


def correlation(C):
    """that calculates a correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")
    if C.shape[1] != C.shape[0]:
        raise ValueError('C must be a 2D square matrix')
    v = np.diag(1 / np.sqrt(np.diag(C)))
    correlation = np.dot(np.dot(v, C), v)
    return correlation
