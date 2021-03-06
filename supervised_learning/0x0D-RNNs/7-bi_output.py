#!/usr/bin/env python3
""" bidirectional Cell """

import numpy as np


class BidirectionalCell():
    """class bidirectionalcell"""
    def __init__(self, i, h, o):
        """ initialize class constructor """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=((2 * h), o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """represents a bidirectional cell of an RNN"""
        hh = np.concatenate((h_prev, x_t), axis=1)
        hs = np.tanh(np.dot(hh, self.Whf) + self.bhf)
        return hs

    def backward(self, h_next, x_t):
        """calculates the hidden state in the backward
        direction for one time step"""
        hh = np.concatenate((h_next, x_t), axis=1)
        hs = np.tanh(np.dot(hh, self.Whb) + self.bhb)
        return hs

    def predict_softmax(self, x):
        """function that performs softmax function"""
        exp_scores = np.exp(x)
        return exp_scores/np.sum((exp_scores), axis=1, keepdims=True)

    def output(self, H):
        """ calculates all outputs for the RNN """
        t, m, h = H.shape
        Y = []

        for t_i in range(t):
            y = np.dot(H[t_i], self.Wy) + self.by
            y = self.predict_softmax(y)
            Y.append(y)
        Y = np.array(Y)
        return Y
