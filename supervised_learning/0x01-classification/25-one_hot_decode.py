#!/usr/bin/env python3
""" one hot encode """
import numpy as np


def one_hot_decode(one_hot):
    """converts a numeric label vector into a one-hot matrix"""
    if len(one_hot.shape) == 2 and type(one_hot) == np.ndarray:
        return np.argmax(one_hot.T, axis=1)
    return None
