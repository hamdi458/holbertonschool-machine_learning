#!/usr/bin/env python3
"""Epsilon Greedy"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """uses epsilon-greedy to determine the next action"""

    e = np.random.uniform(0, 1)

    if e > epsilon:
        return np.argmax(Q[state, :])
    else:
       return np.random.randint(0, 3, None)
