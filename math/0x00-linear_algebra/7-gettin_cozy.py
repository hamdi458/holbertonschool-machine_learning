#!/usr/bin/env python3
"""fn that concatenates two matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """fn that concatenates two matrices"""
    mat = []
    if axis == 1 and len(mat1) == len(mat2):
        for i in range(len(mat1)):
            linge = []
            linge = mat1[i]+mat2[i]
            mat.append(linge)
        return mat
    elif axis == 0 and len(mat1[0]) == len(mat2[0]):
        mat = mat1 + mat2
        return mat
    else:
        return None
