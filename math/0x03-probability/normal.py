#!/usr/bin/env python3
"""loi de Normal"""


class Normal:
    """class normal"""
    def __init__(self, data=None, mean=0., stddev=1.):
        self.data = data
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 3:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data)/len(data)
            som = 0
            for item in data:
                som += (item - self.mean) ** 2
            self.stddev = (som / len(data)) ** 0.5

    def z_score(self, x):
        """calcul pdf"""
        π = 3.1415926536
        e = 2.7182818285
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """calcul cdf"""
        e = 2.7182818285
        π = 3.1415926536
        return z * self.stddev + self.mean

    def pdf(self, x):
        """calcul pdf"""
        e = 2.7182818285
        π = 3.1415926536
        first = 1 / (self.stddev * (2 * π)**0.5)
        return first * e ** (-0.5*((x - self.mean) / self.stddev)**2)

    def cdf(self, x):
        """calcul cdf"""
        π = 3.1415926536
        e = 2.7182818285
        return (1/(2 * π) ** 0.5) * e ** (- x ** 2 / 2)
