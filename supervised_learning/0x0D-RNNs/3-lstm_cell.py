#!/usr/bin/env python3
"""Class GRUCell"""
import numpy as np


class LSTMCell():
    def __init__(self, i, h, o):
        """initialize class constructor"""
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """function that performs Sigmoid"""
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def predict_softmax(self, x):
        """function that performs softmax function"""
        exp_scores = np.exp(x)
        return exp_scores/np.sum((exp_scores), axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """ performs forward propagation for one time step"""
        hh1 = np.concatenate((h_prev, x_t), axis=1)
        s1_U = self.sigmoid(np.dot(hh1, self.Wf) + self.bf)
        s2_U = self.sigmoid(np.dot(hh1, self.Wu) + self.bu)
        tan1_C = np.tanh(np.dot(hh1, self.Wc) + self.bc)
        cs = s1_U * c_prev + s2_U * tan1_C
        s3_O = self.sigmoid(np.dot(hh1, self.Wo) + self.bo)
        hs = s3_O * np.tanh(cs)
        y = np.dot(hs, self.Wy) + self.by
        y = self.predict_softmax(np.dot(hs, self.Wy) + self.by)
        return hs, cs, y
