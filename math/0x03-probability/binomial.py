#!/usr/bin/env python3
"""loi de poisson"""


class Binomial:
    """class Binominal"""
    def __init__(self, data=None, n=1, p=0.5):
        self.data = data
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 3:
                raise ValueError("data must contain multiple values")
            som = 0
            mean = sum(data) / len(data)
            for item in data:
                som += (item - mean) ** 2
            variance = som/len(data)
            self.p = (1-variance/mean)
            self.n = round(mean/self.p)
            self.p = mean/self.n

    def f(self, n):
        """calcul fact"""
        r = 1
        for i in range(1, int(n) + 1):
            r *= i
        return r

    def pmf(self, k):
        """calcul pmf"""
        k = int(k)
        if k < 0:
            return 0
        q = 1 - self.p
        a = self.f(self.n) / (self.f(self.n - k) * self.f(k))
        b = (q ** (self.n-k))
        return a * (self.p ** k) * b
