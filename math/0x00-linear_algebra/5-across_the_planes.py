#!/usr/bin/env python3
'''add matrices'''


def add_matrices2D(mat1, mat2):
    """add 2 matrices"""
    if len(mat2) != len(mat1) or len(mat1[0]) != len(mat2[0]):
        return None
    matrice = []
    for i in range(len(mat1)):
        ligne = []
        for j in range(len(mat1[i])):
            ligne.append(mat2[i][j]+mat1[i][j])
        matrice.append(ligne)
    return(matrice)
