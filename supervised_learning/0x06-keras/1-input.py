#!/usr/bin/env python3
"""function def build_model(nx, layers, activations, lambtha, keep_prob):
that builds a neural network with the Keras library:"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *

from tensorflow.keras.layers import *


def build_model(nx, layers, activations, lambtha, keep_prob):
    x = tf.keras.Input(shape=(nx,))
    for i in range(len(layers)):
        if i == 0:
            y = (Dense(layers[i], activation=activations[i],
                 kernel_regularizer=tf.keras.regularizers.l2(lambtha)))(x)
        else:
            y = (Dense(layers[i], activation=activations[i],
                 kernel_regularizer=tf.keras.regularizers.l2(lambtha)))(y)
        if i < len(layers) - 1:
            y = (Dropout(1-keep_prob))(y)
    model = tf.keras.Model(x, y)
    return model
