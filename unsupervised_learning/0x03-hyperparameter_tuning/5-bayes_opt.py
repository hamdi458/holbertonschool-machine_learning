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
        tab = []

        for i in range(iterations):
            suiv = self.acquisition()[0]
            if suiv in tab:
                break
            suiv_Y = self.f(suiv)
            self.gp.update(suiv, suiv_Y)
            tab.append(suiv)

        if self.minimize is False:
            idx = np.argmax(self.gp.Y)
        else:
            idx = np.argmin(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt