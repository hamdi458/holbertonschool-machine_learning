#!/usr/bin/env python3
"""function def dropout_create_layer(prev, n, activation, keep_prob):
that creates a layer of a neural network using dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates a layer of a neural network using dropout"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(units=n, activation=activation, name='layer',
                            kernel_initializer=init,
                            kernel_regularizer=reg)
    return layer(prev)
