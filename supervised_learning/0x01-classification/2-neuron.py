#!/usr/bin/env python3
"""defines a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """class Neuron"""
    def __init__(self, nx):
        if type(nx) is not int:
            raise ValueError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ get a """
        return self.__W

    @property
    def b(self):
        """ get b """
        return self.__b

    @property
    def A(self):
        """get a"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        nx = (X.shape[0])
        m = (X.shape[1])
        Z = np.dot(self.W, X) + self.b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
