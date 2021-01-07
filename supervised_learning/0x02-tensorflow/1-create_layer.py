#!/usr/bin/env python3
"""tensor output of the layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """the tensor output of the layer"""
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
            units=n,
            kernel_initializer=initialize, activation=activation, name="layer")
