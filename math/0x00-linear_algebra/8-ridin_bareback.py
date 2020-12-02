#!/usr/bin/env python3
"""multiplication matrice"""


def mat_mul(mat1, mat2):
    """multiplication matrice"""
    if len(mat1[0]) != len(mat2):
        return None
    matrice = []
    for i in range(len(mat1)):
        ligne = []
        for k in range(len(mat2[0])):
            s = 0
            for j in range(len(mat1[0])):
                s = s + mat1[i][j] * mat2[j][k]
            ligne.append(s)
        matrice.append(ligne)
    return matrice
