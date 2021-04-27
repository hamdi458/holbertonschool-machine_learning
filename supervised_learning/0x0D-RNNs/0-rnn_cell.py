#!/usr/bin/env python3
"""RNNCell  class"""
import numpy as np


class RNNCell:
    def __init__(self, i, h, o):
        """initialize"""
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def predict_softmax(self, x):
        """function that performs softmax function"""
        exp_scores = np.exp(x)
        return exp_scores/np.sum((exp_scores), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ performs forward propagation for one time step"""
        hh = np.concatenate((h_prev, x_t), axis=1)
        hs = np.tanh(np.dot(hh, self.Wh) + self.bh)
        os = np.dot(hs, self.Wy) + self.by
        y = self.predict_softmax(os)
        return hs, y
