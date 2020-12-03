#!/usr/bin/env python3
"""addition, subtraction, multiplication, and division"""


def np_elementwise(mat1, mat2):
    """addition, subtraction, multiplication, and division"""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return (add, sub, mul, div)
