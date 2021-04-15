#!/usr/bin/env python3
"""class BayesianOptimization"""
import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """class BayesianOptimization that performs Bayesian optimization
    on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        initiallize
        """

        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples, retstep=False)
        self.X_s = X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """that calculates the next best sample location"""

        mu_sample, sig_sample = self.gp.predict(self.X_s)

        if self.minimize:
            Y_sample = np.min(self.gp.Y)
            im = Y_sample - mu_sample - self.xsi
        else:
            Y_sample = np.max(self.gp.Y)
            im = mu_sample - Y_sample - self.xsi

        with np.errstate(divide='ignore'):
            Z = im / sig_sample
            EI = (im * norm.cdf(Z)) + (sig_sample * norm.pdf(Z))
            EI[sig_sample == 0.0] = 0.0

        return self.X_s[np.argmax(EI)], EI

    def optimize(self, iterations=100):
        """optimizes the black-box function"""
        X_aux = []

        for _ in range(iterations):
            Xop, EI = self.acquisition()
            if Xop in X_aux:
                break
            Yop = self.f(Xop)
            self.gp.update(Xop, Yop)
            X_aux.append(Xop)

        if self.minimize:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)

        Xop = self.gp.X[index]
        Yop = self.gp.Y[index]

        return Xop, Yop
