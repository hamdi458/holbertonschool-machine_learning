#!/usr/bin/env python3
"""calculates the integral of a polynomial"""


def poly_derivative(poly, c=0):
    """calculates the integral of a polynomial"""
    if isinstance(poly, list) == 0 or poly == []:
        return None
    if len(poly) == 1:
        return [0]
    new_poly = [0]
    j = 0
    for item in poly:
        j = j + 1
        x = item / j
        if int(x) == x:
            new_poly.append((int)(x))
        else:
            new_poly.append(x)
    return new_poly
