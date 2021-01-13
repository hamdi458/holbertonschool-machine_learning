#!/usr/bin/env python3
"""function def batch_norm(Z, gamma, beta, epsilon):
that normalizes an unactivated output of a neural network
using batch normalization:"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of a neural network
    using batch normalization"""
    u = 1 / Z.shape[0] * np.sum(Z)
    d = 1/Z.shape[0] * np.sum((Z - u)**2)
    z_norm = (Z - u) / (d + epsilon)**0.5
    z = gamma * z_norm + beta
    return z
