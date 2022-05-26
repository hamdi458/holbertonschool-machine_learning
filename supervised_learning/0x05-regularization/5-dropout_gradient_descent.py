#!/usr/bin/env python3
"""fun def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
that updates the weights of a neural network with Dropout
regularization using gradient descent:"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates the weights of a neural network with Dropout"""
    m = Y.shape[1]
    weights_copy = weights.copy()

    for layer_index in range(L, 0, -1):
        A = cache["A" + str(layer_index)]

        if layer_index == L:
            dz = A - Y
        if layer_index>1:
            dz = np.multiply(
                np.dot(weights_copy["W" + str(layer_index + 1)].T, dz),
                (1 - np.power(A, 2)),
            )
            dz = (dz * cache["D" + str(layer_index)]) / keep_prob

        dw = 1 / m * np.dot(dz, cache["A" + str(layer_index - 1)].T)
        db = 1 / m * np.sum(dz, axis=1, keepdims=True)

        weights["W" + str(layer_index)] = weights[
            "W" + str(layer_index)] - (alpha * dw)
        weights["b" + str(layer_index)] = weights[
            "b" + str(layer_index)] - (alpha * db)