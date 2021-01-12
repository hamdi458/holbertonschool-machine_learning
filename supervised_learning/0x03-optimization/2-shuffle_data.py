#!/usr/bin/env python3
"""
function def shuffle_data(X, Y):
that shuffles the data points in two matrices the same way:"""

import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""
    i = np.random.permutation(np.arange(X.shape[0]))
    return X[i], Y[i]
