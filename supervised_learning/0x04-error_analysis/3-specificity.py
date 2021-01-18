#!/usr/bin/env python3
"""function def specificity(confusion):
that calculates the specificity for each class in a confusion matrix:"""


def specificity(confusion):
    """calculates the specificity for each class in a confusion matrix"""
    x = 0
    spec = []
    for x in range(confusion.shape[0]):
        ffp = 0
        ttn = 0
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[0]):
                if i != x and j != x:
                    ttn += confusion[i, j]
                elif x == j:
                    ffp += confusion[i, j]
        spec.append(ttn/(ttn+ffp-confusion[x, x]))
    return spec
