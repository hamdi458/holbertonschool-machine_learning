#!/usr/bin/env python3
"""function def build_model(nx, layers, activations, lambtha, keep_prob):
that builds a neural network with the Keras library"""

import tensorflow.keras as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    model = tf.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(tf.layers.Dense(layers[i], input_dim=nx,
                      activation=activations[i],
                      kernel_regularizer=tf.regularizers.l2(lambtha)))
        else:
            model.add(tf.layers.Dense(layers[i], activation=activations[i],
                      kernel_regularizer=tf.regularizers.l2(lambtha)))
        if i < len(layers) - 1:
            model.add(tf.layers.Dropout(1-keep_prob))
    return model
