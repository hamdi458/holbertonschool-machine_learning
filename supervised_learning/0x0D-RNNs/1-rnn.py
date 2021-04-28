#!/usr/bin/env python3
"""Recurrent Neural Network"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """performs forward propagation for a simple RNN"""
    t, m, i = X.shape
    H = []
    Y = []
    h = h_0
    H.append(h_0)
    for t_i in range(t):
        h, y = rnn_cell.forward(h, X[t_i])
        H.append(h)
        Y.append(y)
    H = np.array(H)
    Y = np.array(Y)
    return H, Y
