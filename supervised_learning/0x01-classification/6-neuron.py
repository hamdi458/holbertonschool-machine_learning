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

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        q = - (1 / m)
        e = np.multiply(1 - Y, np.log(1.0000001 - A))
        cost = q * np.sum(np.multiply(Y, np.log(A)) + e)
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        a = self.forward_prop(X)
        pred = np.round(a)
        return pred.astype(np.int), self.cost(Y, a)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        self.__W = self.__W - (((1 / m) * np.sum((A - Y) * X, axis=1)) * alpha)
        self.__b = self.__b - alpha * np.sum(A - Y) / m

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """evaluation of the training data after iterations of training"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for it in range(iterations):
            self.gradient_descent(X, Y, self.__A, alpha)
            self.forward_prop(X)
        return self.evaluate(X, Y)
