#!/usr/bin/env python3
""" function def dropout_forward_prop(X, weights, L, keep_prob):
that conducts forward propagation using Dropout:"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout"""
    cache = {}
    cache['A0'] = X
    for i in range(L):
        b = weights["b" + str(i+1)]
        z = np.dot(weights['W'+str(i+1)], cache["A"+str(i)])+b
        if i == L - 1:
            g = np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)), axis=0)
        else:
            t = (2 / (1 + np.exp(-2 * z))) - 1
            d = np.random.binomial(1, keep_prob, size=t.shape)
            g = t * d / keep_prob
            cache["D"+str(i+1)] = d
        cache["A"+str(i+1)] = g
    return cache
