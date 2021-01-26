#!/usr/bin/env python3
"""function def one_hot(labels, classes=None):
that converts a label vector into a one-hot matrix:"""

import tensorflow.keras as k


def one_hot(labels, classes=None):
    """converts a label vector into a one-hot matrix With keras"""
    encoded = k.utils.to_categorical(labels)
    return(encoded)
