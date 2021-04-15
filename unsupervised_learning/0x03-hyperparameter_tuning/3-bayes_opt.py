#!/usr/bin/env python3
"""class BayesianOptimization"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """class BayesianOptimization that performs Bayesian optimization
    on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, vac_samples,
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
