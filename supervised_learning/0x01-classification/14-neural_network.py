#!/usr/bin/env python3
"""defines a neural network with one hidden layer"""
import numpy as np


class NeuralNetwork:
    """class NeuralNetwork"""
    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """The weights vector for the hidden layer. Upon instantiation"""
        return self.__W1

    @property
    def b1(self):
        """The bias for the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """The activated output for the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """The weights vector for the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """The bias for the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """The activated output for the output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """forward propagation"""
        Z1 = np.dot(self.__W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """calcul cost"""
        m = Y.shape[1]
        q = - (1 / m)
        e = np.multiply(1 - Y, np.log(1.0000001 - A))
        cost = q * np.sum(np.multiply(Y, np.log(A)) + e)
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        a, b = self.forward_prop(X)
        pred = np.round(b)
        return pred.astype(np.int), self.cost(Y, b)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """gradient descent"""
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = 1 / m * np.dot(dz2, A1.T)
        db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.dot(self.__W2.T, dz2) * (A1 - A1 ** 2)
        db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
        dw1 = 1 / m * np.dot(dz1, X.T)
        self.__W1 = self.__W1 - alpha * dw1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dw2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """evaluation of the training data after iterations"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for it in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        return self.evaluate(X, Y)