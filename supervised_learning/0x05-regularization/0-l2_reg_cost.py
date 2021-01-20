#!/usr/bin/env python3
"""function def l2_reg_cost(cost, lambtha, weights, L, m):
that calculates the cost of a neural network with L2 regularization:"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """that calculates the cost of a neural network with L2 regularization"""
    W1 = weights["W1"]
    W2 = weights["W2"]
    W3 = weights["W3"]
    L2_regularization_cost = (np.sum(np.linalg.norm(W1)) + np.sum(np.linalg.norm(W2)) + np.sum(np.linalg.norm(W3)))*(lambtha/(2*m))
    cost = cost + L2_regularization_cost
    return cost
