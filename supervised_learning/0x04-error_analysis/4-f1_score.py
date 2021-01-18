#!/usr/bin/env python3
""" function def f1_score(confusion):
that calculates the F1 score of a confusion matrix"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the F1 score of a confusion matrix"""
    preci = precision(confusion)
    sensi = sensitivity(confusion)
    return 2 * preci * sensi / (preci + sensi)
