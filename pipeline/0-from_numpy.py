#!/usr/bin/env python3
""" From Numpy
"""
import numpy as np
import pandas as pd
import string


def from_numpy(array):
    """ creates a pd.DataFrame from a np.ndarray"""
    index_values = list(string.ascii_lowercase)
    index = []
    for i in range(array.shape[1]):
        index.append(index_values[i])
    df = pd.DataFrame(data=array,
                      columns=index)
    return df
