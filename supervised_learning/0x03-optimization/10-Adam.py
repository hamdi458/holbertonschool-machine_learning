#!/usr/bin/env python3
"""function def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
that creates the training operation for a neural network in tensorflow
using the Adam optimization algorithm:"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    minimize = tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        use_locking=False,
        name='Adam')
    return minimize.minimize(loss)
