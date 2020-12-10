#!/usr/bin/env python3
"""calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    if isinstance(poly, list) == 0 or poly == []:
        return None
    if len(poly) == 1:
        return [0]
    poly.pop(0)
    new_poly = []
    j = 0
    for i in range(0, len(poly)):
        new_poly.append(poly[i] * (j + 1))
        j = j + 1
    return new_poly
