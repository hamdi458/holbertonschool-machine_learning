#!/usr/bin/env python3
""" create layer"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ create layer"""
    tr = tf.contrib.layers.l2_regularizer(lambtha)
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=kernel, name="layer",
                            kernel_regularizer=tr,
                            )
    return layer(prev)
