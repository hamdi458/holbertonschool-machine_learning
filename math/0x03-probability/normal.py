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
            if type(data) is not list:
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

    def erf(self, x):
        """ erf function"""
        π = 3.1415926536
        a = (2 / π ** (1 / 2))
        b = ((x ** 3) / 3)
        c = ((x ** 7) / 42)
        return a * (x - b + ((x ** 5) / 10) - c + ((x ** 9) / 216))

    def cdf(self, x):
        """ cdf fn"""
        a = (1 + self.erf((x - self.mean) / (self.stddev * (2 ** (1 / 2)))))
        return (1 / 2) * a
