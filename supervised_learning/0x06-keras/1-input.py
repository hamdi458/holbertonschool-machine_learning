#!/usr/bin/env python3
"""function def build_model(nx, layers, activations, lambtha, keep_prob):
that builds a neural network with the Keras library:"""

import tensorflow.keras as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    x = tf.Input(shape=(nx,))
    for i in range(len(layers)):
        if i == 0:
            y = (tf.layers.Dense(layers[i], activation=activations[i],
                 kernel_regularizer=tf.regularizers.l2(lambtha)))(x)
        else:
            y = (tf.layers.Dense(layers[i], activation=activations[i],
                 kernel_regularizer=tf.regularizers.l2(lambtha)))(y)
        if i < len(layers) - 1:
            y = (tf.layers.Dropout(1-keep_prob))(y)
    model = tf.Model(x, y)
    return model
