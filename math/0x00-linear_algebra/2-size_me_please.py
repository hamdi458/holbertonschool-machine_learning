#!/usr/bin/env python3
"""wiw"""


def matrix_shape(matrix):
    """wiw"""    
    mat_new = []
    while isinstance(matrix, list):
        mat_new.append(len(matrix))
        matrix = matrix[0]
    return mat_new
