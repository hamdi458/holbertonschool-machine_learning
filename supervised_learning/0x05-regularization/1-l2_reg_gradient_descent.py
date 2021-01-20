#!/usr/bin/env python3
"""function def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
updates the parameters using gradient descent with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the parameters using gradient descent with L2 regularization"""
    weights2 = weights.copy()
    m = Y.shape[1]
    dz = cache["A"+str(L)] - Y
    for i in range(L, 0, -1):
        c = cache["A" + str(i - 1)]
        b = "b" + str(i)
        w = "W" + str(i)
        dw = (1 / m) * np.matmul(dz, c.T)+((lambtha / m) * weights[w])
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        weights[w] = weights[w] - alpha * dw
        weights[b] = weights[b] - alpha * db
        dz = np.matmul(weights2[w].T, dz) * (1 - np.power(c, 2))
    return weights
