#!/usr/bin/env python3
"""The Viretbi Algorithm"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    calculates the most likely sequence of hidden states
    for a hidden markov model
    """
    F = np.zeros((Emission.shape[0], Observation.shape[0]))
    all_prev_states = np.zeros((Emission.shape[0], Observation.shape[0]))
    ob_ind = Observation[0]
    F[:, 0] = np.multiply(Initial.T, Emission[:, ob_ind])
    all_prev_states[:, 0] = 0
    for i in range(1, Observation.shape[0]):
        for j in range(Emission.shape[0]):
            ob_ind = Observation[i]
            X = Transition[:, j] * F[:, i - 1]
            alpha = Emission[j, ob_ind]
            F[j, i] = np.max(alpha * X)
            all_prev_states[j, i] = np.argmax(alpha * X)
    P = F[:, Observation.shape[0] - 1].max()
    state = int(np.argmax(F[:, Observation.shape[0] - 1]))
    state_sequence = []
    state_sequence.append(state)
    for i in range(Observation.shape[0] - 1, 0, -1):
        state_sequence.append(int(all_prev_states[state, i]))
        state = int(all_prev_states[state, i])
    state_sequence.reverse()

    return state_sequence, P
