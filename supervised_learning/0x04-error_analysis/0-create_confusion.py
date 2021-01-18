#!/usr/bin/env python3
"""function def create_confusion_matrix(labels, logits):
that creates a confusion matrix:"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix"""
    classes = labels.shape[1]
    cm = np.zeros((classes, classes))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] == 1:
                x = j
            if logits[i, j] == 1:
                y = j
        cm[x, y] = cm[x, y] + 1
    return cm
