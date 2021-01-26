#!/usr/bin/env python3
"""function def one_hot(labels, classes=None):
that converts a label vector into a one-hot matrix:"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from tensorflow.keras.utils import to_categorical


def one_hot(labels, classes=None):
    """converts a label vector into a one-hot matrix With keras"""
    encoded = to_categorical(labels)
    return(encoded)
