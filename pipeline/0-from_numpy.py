#!/usr/bin/env python3
""" From Numpy
"""
import numpy as np
import pandas as pd
import string


def from_numpy(array):
    """ creates a pd.DataFrame from a np.ndarray"""
    index_values = list(string.ascii_lowercase)
    comumn_number = array.shape[0]
    column_values = []
    for i in range(comumn_number):
        column_values.append(i)
    index = []
    for i in range(array.shape[1]):
        index.append(index_values[i])
    df = pd.DataFrame(data=array,
                      index=column_values,
                      columns=index)
    return df
