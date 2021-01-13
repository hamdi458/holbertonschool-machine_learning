#!/usr/bin/env python3
"""function def create_RMSProp_op(loss, alpha, beta2, epsilon):
that creates the training operation for a neural network in tensorflow
using the RMSProp optimization algorithm:"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """updates a variable using the RMSProp optimization algorithm"""
    return tf.train.RMSPropOptimizer(alpha,
                                     beta2,
                                     epsilon=epsilon).minimize(loss)
