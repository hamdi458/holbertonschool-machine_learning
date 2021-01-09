#!/usr/bin/env python3
""" function def normalization_constants(X):
that calculates the normalization (standardization)
constants of a matrix"""

import numpy as np


def normalization_constants(X):
    """calculates the normalization"""
    return np.mean(X, axis=0), np.std(X, axis=0)
