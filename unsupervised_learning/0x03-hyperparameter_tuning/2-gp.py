#!/usr/bin/env python3
"""Class GaussianProcess"""
import numpy as np


class GaussianProcess():
    """represents a noiseless 1D Gaussian process    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """initiallize"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """calculates the covariance kernel matrix between two matrices"""
        s = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 \
            * np.dot(X1, X2.T)
        K = self.sigma_f**2 * np.exp(-0.5 / self.l**2 * s)

        return K

    def predict(self, X_s):
        """predicts the mean and standard deviation of points
        in a Gaussian process"""
        K_inv = np.linalg.inv(self.K)
        Ks = self.kernel(self.X, X_s)
        Kss = self.kernel(X_s, X_s)

        s = Ks.T.dot(K_inv).dot(self.Y)
        mu = np.reshape(s, -1)
        cs = Kss - Ks.T.dot(K_inv).dot(Ks)
        sig = np.diagonal(cs)

        return mu, sig

    def update(self, X_new, Y_new):
        """Updates the public instance attributes X, Y, and k"""
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
