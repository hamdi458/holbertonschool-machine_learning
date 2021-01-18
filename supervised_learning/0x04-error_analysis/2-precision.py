#!/usr/bin/env python3
"""function def precision(confusion):
that calculates the precision for each class in a confusion matrix:"""
import numpy as np


def precision(confusion):
    """calculates the precision for each class in a confusion matrix"""
    tp = []
    for i in range(confusion.shape[0]):
        tp.append(confusion[i, i])
    autr = np.sum(confusion, axis=0)
    return tp / autr
