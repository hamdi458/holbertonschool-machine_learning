#!/usr/bin/env python3
"""function def batch_norm(Z, gamma, beta, epsilon):
that normalizes an unactivated output of a neural network
using batch normalization:"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of a neural network
    using batch normalization"""
    u = np.mean(Z, axis=0)
    variance = np.var((Z - u), axis=0)
    z_norm = (Z - u) / np.sqrt(variance + epsilon)
    z = gamma * z_norm + beta
    return z
