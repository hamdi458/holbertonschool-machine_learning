#!/usr/bin/env python3
"""function def sensitivity(confusion):
that calculates the sensitivity for each class in a confusion matrix:"""

import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each class in a confusion matrix"""

    sensitivitynb = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        tp = 0
        fn = 0
        for j in range(confusion.shape[0]):
            if i == j:
                tp = tp + confusion[i, j]
            else:
                fn = fn + confusion[i, j]
        sensitivitynb[i] = tp/(tp + fn)
    return sensitivitynb
