#!/usr/bin/env python3
"""determines the steady state probabilities of a regular markov chain"""
import numpy as np


def regular(P):
    """determines the steady state probabilities of a regular markov chain"""
    n = P.shape[0]

    s = np.ones((1, n)) / n
    for i in range(n):
        for j in range(n):
            if P[i, j] <= 0:
                return None
    while (s != np.dot(s, P)).any():
        s = np.dot(s, P)
    return s
