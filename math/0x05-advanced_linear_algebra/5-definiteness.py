#!/usr/bin/env python3
"""Advanced linear algebra module"""
import numpy as np


def definiteness(matrix):
    """calculates the definiteness of a matrix"""

    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')

    if not np.array_equal(matrix.T, matrix):
        return None
    try:
        if np.all(np.linalg.eig(matrix)[0] > 0):
            return("Positive definite")
        elif np.all(np.linalg.eig(matrix)[0] < 0):
            return("negative definite")
        elif np.all(np.linalg.eig(matrix)[0] >= 0):
            return("Positive semi-definite")
        elif np.all(np.linalg.eig(matrix)[0] <= 0):
            return("negative semi-definite")
        else:
            return("Indefinite")
    except Exception:
        return None