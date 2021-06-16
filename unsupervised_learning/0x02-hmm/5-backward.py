#!/usr/bin/env python3
"""backward algorithm markov model"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """fn that performs the backward algorithm for a hidden markov model"""
    F = np.zeros((Emission.shape[0], Observation.shape[0]))
    F[:, - 1] = np.ones(Emission.shape[0])
    ob_ind = Observation[0]
    Initial = np.multiply(Initial.T, Emission[:, ob_ind])

    for i in range(Observation.shape[0]-2, -1, -1):
        for j in range(Emission.shape[0]):
            ob_ind = Observation[i+1]
            X = Transition[j] * F[:, i+1]
            alpha = Emission[:, ob_ind]
            F[j:, i] = np.sum(np.multiply(alpha, X))

    P = np.sum(Initial * F[:, 0])
    return P, F
