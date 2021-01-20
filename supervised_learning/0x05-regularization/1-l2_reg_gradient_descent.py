#!/usr/bin/env python3
import numpy as np
def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    weights2 = weights.copy()
    m = Y.shape[1]
    W1 = weights["W1"]
    b1 = weights["b1"]
    W2 = weights["W2"]
    b2 = weights["b2"]
    W3 = weights["W3"]
    b3 = weights["b3"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    dz = A3 - Y
    for i in range(L, 0, -1):
        c = cache["A" + str(i - 1)]
        b = "b" + str(i)
        w = "W" + str(i)
        dw = (1 / m) * np.matmul(dz, c.T)+ ((lambtha / m) * weights[w])
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        weights[w] = weights[w] - alpha * dw
        weights[b] = weights[b] - alpha * db
        dz = np.matmul(weights2[w].T, dz) * (1 - np.power(c, 2))
    return weights