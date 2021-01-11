#!/usr/bin/env python3
""" create layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """ create layer"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=kernel, name="layer",
                            )
    return layer(prev)