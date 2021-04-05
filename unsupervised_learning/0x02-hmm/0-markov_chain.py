#!/usr/bin/env python3
"""markov prob"""
import numpy as np


def markov_chain(P, s, t=1):
    """determines the probability of a markov chain being
    in a particular state after a specified number of iterations"""
    for i in range(t):
        s = np.dot(s, P)
    return s
