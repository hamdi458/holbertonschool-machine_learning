#!/usr/bin/env python3
"""
function def normalize(X, m, s): that normalizes (standardizes) a matrix
"""

import numpy as np


def normalize(X, m, s):
    normalize = (X - m) / s
    return normalize
