#!/usr/bin/env python3
"""fun def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
that updates the weights of a neural network with Dropout
regularization using gradient descent:"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates the weights of a neural network with Dropout"""
    wht = weights.copy()
    m = Y.shape[1]
    dz = cache["A"+str(L)] - Y
    for i in range(L, 0, -1):
        dw = 1 / m * np.dot(dz, cache["A"+str(i-1)].T)
        db = 1 / m * np.sum(dz, axis=1, keepdims=True)
        weights["W"+str(i)] = weights["W"+str(i)]-(dw*alpha)
        weights["b"+str(i)] = weights["b"+str(i)]-(db*alpha)
        w = (cache["A"+str(i-1)] - cache["A"+str(i-1)] ** 2)
        qqq = (1 - np.power(cache["A"+str(i-1)], 2))
        if i > 1:
            a = cache['D' + str(i-1)] / keep_prob
            dz = np.dot(wht['W' + str(i)].T, dz) * qqq * a

