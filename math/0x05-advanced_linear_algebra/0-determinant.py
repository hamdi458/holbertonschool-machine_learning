#!/usr/bin/env python3
"""calculates the determinant of a matrix"""


def construire_mat(mat, i):
    """
    construire a new matrix
    """
    new_mat = []
    for r in range(1, len(mat)):
        wiw = []
        for j in range(len(mat[0])):
            if j != i:
                wiw.append(mat[r][j])
        new_mat.append(wiw)
    return new_mat


def determinant(matrix):
    """
    matrix is a list of lists whose determinant should be calculated
    If matrix is not a list of lists, raise a TypeError with
        the message matrix must be a list of lists
    If matrix is not square, raise a ValueError with
        the message matrix must be a square matrix
    The list [[]] represents a 0x0 matrix
    Returns: the determinant of matrix
    """

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    for r in matrix:
        if not isinstance(r, list):
            raise TypeError("matrix must be a list of lists")

    for r in matrix:
        if len(r) != len(matrix):
            raise ValueError('matrix must be a square matrix')

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for i, j in enumerate(matrix[0]):

        new_mat = construire_mat(matrix, i)
        if i % 2 == 1:
            det += j * (-1) ** i * determinant(new_mat)
        else:
            det += j * determinant(new_mat)
    return det
