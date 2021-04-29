#!/usr/bin/env python3
"""Class GRUCell"""
import numpy as np


class GRUCell():
    def __init__(self, i, h, o):
        """initialize class constructor"""
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """function that performs Sigmoid"""
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def predict_softmax(self, x):
        """function that performs softmax function"""
        exp_scores = np.exp(x)
        return exp_scores/np.sum((exp_scores), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ performs forward propagation for one time step"""
        hh1 = np.concatenate((h_prev, x_t), axis=1)
        s1 = self.sigmoid(np.dot(hh1, self.Wr) + self.br)
        s2 = self.sigmoid(np.dot(hh1, self.Wz) + self.bz)

        hh2 = np.concatenate(((s1 * h_prev),x_t), axis=1)
        tan = np.tanh(np.matmul(hh2, self.Wh) + self.bh)
        hs = (1 - s2) * h_prev + s2 * tan
        y = np.dot(hs, self.Wy) + self.by
        y = self.predict_softmax(y)
        return hs, y
