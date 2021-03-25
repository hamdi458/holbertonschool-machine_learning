#!/usr/bin/env python3
"""marginal probability"""
import numpy as np


def marginal(x, n, P, Pr):
    """Function that calculates the marginal probability of obtaining the data"""

    if not isinstance(n, int) or (n <= 0):
        raise ValueError('n must be a positive integer')

    if not isinstance(x, int) or (x < 0):
        err = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(err)

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError('All values in P must be in the range [0, 1]')

    if not isinstance(Pr, np.ndarray) or (P.shape != Pr.shape):
        err = 'Pr must be a numpy.ndarray with the same shape as P'
        raise TypeError(err)

    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError('All values in Pr must be in the range [0, 1]')

    E = np.sum(Pr)
    if not np.isclose(E, 1):
        raise ValueError('Pr must sum to 1')

    fa = (np.math.factorial(n))
    d = (np.math.factorial(x) * np.math.factorial(n - x))
    fact = fa / d
    fact *= (np.power(P, x)) * (np.power((1 - P), (n - x)))

    inter = fact * Pr
    return np.sum(inter)
