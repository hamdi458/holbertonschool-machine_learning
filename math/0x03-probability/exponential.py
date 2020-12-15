#!/usr/bin/env python3
"""loi de poisson"""


class Exponential:
    """class exponential"""
    def __init__(self, data=None, lambtha=1.):
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = (float)(lambtha)
        else:
            if type(data) is not list:
                raise ValueError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            lam = len(data)/sum(data)
            self.lambtha = lam

    def pdf(self, x):
        """calcul pdf"""
        e = 2.7182818285
        if x < 0:
            return 0
        return (self.lambtha) * e ** (-self.lambtha * x)

    def cdf(self, x):
        """calcul cdf"""
        e = 2.7182818285
        if x < 0:
            return 0
        return 1 - e ** (-self.lambtha * x)
