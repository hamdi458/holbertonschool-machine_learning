#!/usr/bin/env python3
"""calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if isinstance(poly, list) == 0 or poly == []:
        return None
    if isinstance(C, (float, int)) == 0:
        return None
    if len(poly) == 1 and poly[0] == 0:
        return [C]
    new_poly = [C]
    j = 0
    for item in poly:
        j = j + 1
        x = item / j
        if int(x) == x:
            new_poly.append((int)(x))
        else:
            new_poly.append(x)
    return new_poly
