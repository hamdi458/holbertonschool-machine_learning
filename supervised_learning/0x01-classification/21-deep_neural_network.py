#!/usr/bin/env python3
"""defines a neural network with one hidden layer"""
import numpy as np


class DeepNeuralNetwork:
    """class NeuralNetwork"""
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.weights['b' + str(1)] = np.zeros((layers[0], 1))
        self.weights['W' + str(1)] = np.random.normal(size=(layers[0],
                                                      nx))*np.sqrt(2/nx)
        for i in range(1, len(layers)):
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
            a = np.sqrt(2/layers[i-1])
            la = np.random.normal(size=(layers[i], layers[i-1]))*a
            self.weights['W' + str(i + 1)] = la

    @property
    def L(self):
        """The number of layers in the neural network"""
        return self.__L

    @property
    def cache(self):
        """dictionary to hold all intermediary values of the network"""
        return self.__cache

    @property
    def weights(self):
        """dictionary to hold all weights and biased of the network."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.cache['A0'] = X
        for i in range(self.L):
            b = self.__weights["b" + str(i+1)]
            z = np.dot(self.weights['W'+str(i+1)], self.cache["A"+str(i)])+b
            self.cache["A"+str(i+1)] = 1/(1+np.exp(-z))
        return self.cache["A"+str(self.L)], self.cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        q = - (1 / m)
        e = np.multiply(1 - Y, np.log(1.0000001 - A))
        cost = q * np.sum(np.multiply(Y, np.log(A)) + e)
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        c, c1 = self.forward_prop(X)
        pred = np.round(c)
        return pred.astype(np.int), self.cost(Y, c)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """gradient descent"""
        wht = self.__weights.copy()
        m = Y.shape[1]
        dz = cache["A"+str(self.L)] - Y
        for i in range(self.L, 0, -1):
            dw = 1 / m * np.dot(dz, self.__cache["A"+str(i-1)].T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)
            self.__weights["W"+str(i)] = self.__weights["W"+str(i)]-(dw*alpha)
            self.__weights["b"+str(i)] = self.__weights["b"+str(i)]-(db*alpha)
            w = (self.cache["A"+str(i-1)] - self.cache["A"+str(i-1)] ** 2)
            dz = np.dot(wht["W"+str(i)].T, dz) * w
        return self.__weights
