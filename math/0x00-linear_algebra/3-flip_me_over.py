#!/usr/bin/env python3
"""fn python"""


def matrix_transpose(matrix):
    """transpose"""
    mat = []
    for i in range(len(matrix[0])):
        m = []
        for j in range(len(matrix)):
            m.append(matrix[j][i])
        mat.append(m)
    return mat
