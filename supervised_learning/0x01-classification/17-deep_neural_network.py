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
