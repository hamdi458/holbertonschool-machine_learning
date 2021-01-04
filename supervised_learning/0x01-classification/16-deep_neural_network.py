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
        if type(layers) is not list or layers is None:
            raise TypeError("layers must be a list of positive integers")
        for item in layers:
            if type(item) is not int or item < 1:
                raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        self.weights['b' + str(1)] = np.zeros((layers[0], 1))
        self.weights['W' + str(1)] = np.random.normal(size=(layers[0],
                                                      nx))*np.sqrt(2/nx)
        for i in range(len(layers)):
            if i != 0:
                self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
                a = np.sqrt(2/layers[i-1])
                la = np.random.normal(size=(layers[i], layers[i-1]))*a
                self.weights['W' + str(i + 1)] = la
