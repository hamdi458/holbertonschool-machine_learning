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

    def factoriel(self,n):
        """fact"""
        fact = 1
        for i in range(1, int(n) + 1):
            fact = fact * i
        return fact                 
 
    def pmf(self, k):
        """calcul pmf"""
        k = int(k)
        if k < 0:
            return 0
        q = 1 - self.p
        a = self.factoriel(self.n) / (self.factoriel(self.n - k) * self.factoriel(k))
        b = (self.p** k) * (q ** (self.n-k))
        return comb * b
        