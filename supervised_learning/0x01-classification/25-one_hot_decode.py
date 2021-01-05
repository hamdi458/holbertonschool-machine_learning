#!/usr/bin/env python3
""" one hot encode """
import numpy as np


def one_hot_decode(one_hot):
    """converts a numeric label vector into a one-hot matrix"""
    if type(one_hot) != np.ndarray or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot.T, axis=1)
