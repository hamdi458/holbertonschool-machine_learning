#!/usr/bin/env python3
"""calculates the cofactor of a matrix"""


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


def const_mat_det(mat, i, j):
    """const mat determinant"""
    mat_det = []
    for r in range(len(mat)):
        mat_det_row = []
        for c in range(len(mat)):
            if i != r and j != c:
                mat_det_row.append(mat[r][c])
        if len(mat_det_row) > 0:
            mat_det.append(mat_det_row)
    return(mat_det)


def minor(matrix):
    """that calculates the minor matrix of a matrix
    """
    if matrix == []:
        raise TypeError("matrix must be a list of lists")
    if type(matrix[0]) is not list or type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    elif len(matrix[0]) != len(matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    for i in range(1, len(matrix)):
        if type(matrix[i]) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix[i]) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    minor = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[0])):
            mat_det = const_mat_det(matrix, i, j)
            det = determinant(mat_det)
            row.append(det)
        minor.append(row)
    return minor


def cofactor(matrix):
    """
    Function that calculates the cofactor matrix of a matrix
    """
    pos = 0
    mat_cofactor = minor(matrix)
    for i in range(len(mat_cofactor)):
        for j in range(len(mat_cofactor)):
            pos += 1
            if pos % 2 == 0:
                mat_cofactor[i][j] *= (-1)

    return mat_cofactor
