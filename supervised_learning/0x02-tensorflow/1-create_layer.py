#!/usr/bin/env python3
"""tensor output of the layer"""

import tensorflow as tf

def create_layer(prev, n, activation):
    """the tensor output of the layer"""
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
