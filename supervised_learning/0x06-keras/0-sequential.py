#!/usr/bin/env python3
"""function def build_model(nx, layers, activations, lambtha, keep_prob):
that builds a neural network with the Keras library"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *

from tensorflow.keras.layers import *


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    model = keras.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(Dense(layers[i], input_dim=nx, activation=activations[i],
                      kernel_regularizer=tf.keras.regularizers.l2(lambtha)))
        else:
            model.add(Dense(layers[i], activation=activations[i],
                      kernel_regularizer=tf.keras.regularizers.l2(lambtha)))
        if i < len(layers) - 1:
            model.add(Dropout(1-keep_prob))
    return model
