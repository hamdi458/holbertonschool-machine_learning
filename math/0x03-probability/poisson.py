#!/usr/bin/env python3
"""loi de poisson"""


class Poisson:
    """class poisson"""

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = (float)(lambtha)
        else:
            if type(data) is not list:
                raise ValueError("data must be a list")
            if len(data) < 2 :
                raise ValueError("data must contain multiple values")
            lam = sum(data)/len(data)
            self.lambtha = (float)(lam)