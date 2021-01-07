#!/usr/bin/env python3
"""eturns two placeholders, x and y, for the neural network"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """eturns two placeholders, x and y, for the neural network"""
    x = tf.placeholder("float", name="x")
    y = tf.placeholder("float",x*2, name="y")
    return x, y
