#!/usr/bin/env python3
def matrix_shape(matrix):
    mat_new = []
    l = len(matrix)
    while isinstance(matrix, list):
        mat_new.append(l)
        matrix = matrix[0]
    return mat_new