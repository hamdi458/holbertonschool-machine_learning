#!/usr/bin/env python3
"""
Module for class DeepNeuralNetwork
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing
    binary classification
    """

    def __init__(self, nx, layers):
        """
        Class Constructor
        nx is the number of input features
        layers is a list representing the number of
            nodes in each layer of the network
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or layers == []:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for layer in range(self.__L):
            if layers[layer] < 1:
                raise TypeError("layers must be a list of positive integers")
            if layer == 0:
                layer_prev = nx
            else:
                layer_prev = layers[layer - 1]
            self.__weights['W' + str(layer + 1)] = np.random.randn(
                layers[layer], layer_prev) * np.sqrt(2 / layer_prev)
            self.__weights['b' + str(layer + 1)] = np.zeros((layers[layer], 1))

    @property
    def L(self):
        """ Getter for L """
        return self.__L

    @property
    def cache(self):
        """ Getter for cache """
        return self.__cache

    @property
    def weights(self):
        """ Getter for weights """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        self.__cache["A0"] = X
        for layer in range(self.__L):
            Z = np.matmul(
                self.__weights["W" + str(layer + 1)],
                self.__cache["A" + str(layer)]) + \
                self.__weights["b" + str(layer + 1)]
            self.__cache["A" + str(layer + 1)] = 1 / (1 + np.exp(-Z))
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = -1 / Y.shape[1]
        cost = m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions """
        A = self.forward_prop(X)[0]
        pred = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron """
        m = 1 / Y.shape[1]
        for i in reversed(range(1, self.__L + 1)):
            A_prev = self.__cache["A" + str(i - 1)]
            A = self.__cache["A" + str(i)]
            W = self.__weights["W" + str(i)]
            if i == self.__L:
                dz = A - Y
            else:
                dz = da * (A * (1 - A))
            dw = m * np.matmul(dz, A_prev.T)
            db = m * np.sum(dz, axis=1, keepdims=True)
            da = np.matmul(W.T, dz)
            self.__weights["W" + str(i)] -= (alpha * dw)
            self.__weights["b" + str(i)] -= (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """evaluation of the training data after iterations"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        arrX = []
        arrY = []
        for it in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            j = self.cost(Y, self.cache["A"+str(self.L)])
            if verbose:
                print(f"Cost after {it} iterations: {j}")
                arrX.append(it)
                arrY.append(j)
        if graph:
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.plot(arrX, arrY)
            plt.show()
        return self.evaluate(X, Y)
    def save(self, filename):
        """ Saves the instance object to a file in pickle format """
        if type(filename) != str:
            return None
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """ Loads a pickled DeepNeuralNetwork object """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None