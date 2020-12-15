#!/usr/bin/env python3
"""loi de poisson"""


class Poisson:
    """class poisson"""
    def __init__(self, data=None, lambtha=1.):
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 3:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data)/len(data)

    def pmf(self, k):
        """calcul pmf"""
        π = 3.1415926536
        e = 2.7182818285
        k = (int)(k)
        if k < 0:
            return 0
        x = 1
        for i in range(2, k + 1):
            x *= i
        return((self.lambtha ** k) / x) * e ** (-self.lambtha)

    def cdf(self, k):
        """calcul cdf"""
        k = (int)(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(0, k + 1):
            cdf += self.pmf(i)
        return cdf
