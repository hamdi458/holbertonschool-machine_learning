#!/usr/bin/env python3
"""sigma au carré"""


def summation_i_squared(n):
    """return sum sigma au carré"""
    if n is None or n <= 0:
        return None
    return (int)(n * (n + 1) * (2 * n + 1) / 6)
