#!/usr/bin/env python3
"""Simple Policy function"""
import numpy as np


def policy_gradient(state, weight):
    """function that computes to policy with a weight of a matrix"""
    def softmax_grad(softmax):
        s = softmax.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
    z = state.dot(weight)
    exp = np.exp(z)
    prb = exp / np.sum(exp)
    action = np.argmax(prb)
    dsoftmax = softmax_grad(prb)[action, :]
    dlog = dsoftmax / prb[0, action]
    gradient = state.T.dot(dlog[None, :])
    return (action, gradient)
