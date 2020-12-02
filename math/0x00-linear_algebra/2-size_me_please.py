#!/usr/bin/env python3
"""shape matrice"""


def matrix_shape(matrix):
    """find matrice shape"""    
    mat_new = []
    while isinstance(matrix, list):
        mat_new.append(len(matrix))
        matrix = matrix[0]
    return mat_new
