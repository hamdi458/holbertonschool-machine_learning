#!/usr/bin/env python3
""" absorbing chain """
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing"""
    diag = np.diag(P)
    if not((diag == 1).any()):
        return False
    elif (diag == 1).all():
        return True
    chemin_absor = []
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            if P[i, j] == 1 and i == j:
                chemin_absor.append(i)

    for _ in range(P.shape[0]):
        for i in range(P.shape[0]):
            for j in range(P.shape[0]):
                if (P[i, j] > 0 and not(any(chemin_absor) == i)):
                    if and any(chemin_absor) == j:
                        chemin_absor.append(i)
    if len(chemin_absor) < P.shape[0]:
        return False
    else:
        return True
