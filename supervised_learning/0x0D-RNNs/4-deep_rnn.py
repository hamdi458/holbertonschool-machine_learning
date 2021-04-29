#!/usr/bin/env python3
"""Program that performs forward propagation for a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Function that performs forward propagation for a deep RNN"""
    t, m, i = X.shape
    H = []
    H.append(h_0)
    Y = []
    for t_i in range(t):
        k = 0
        hh = []
        h = X[t_i]
        for i in (rnn_cells):
            h, y = i.forward(H[t_i][k], h)
            hh.append(h)
            k = k + 1
        H.append(hh)
        Y.append(y)
    H = np.array(H)
    Y = np.array(Y)
    return H, Y
