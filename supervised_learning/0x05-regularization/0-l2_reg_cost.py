#!/usr/bin/env python3
"""function def l2_reg_cost(cost, lambtha, weights, L, m):
that calculates the cost of a neural network with L2 regularization:"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """that calculates the cost of a neural network with L2 regularization"""
    sum_w = 0
    for i in range(L):
        sum_w = sum_w + np.linalg.norm(weights['W' + str(i+1)])
    L2_regularization_cost = sum_w*(lambtha/(2*m))
    cost = cost + L2_regularization_cost
    return cost
