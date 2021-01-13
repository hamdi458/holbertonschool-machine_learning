#!/usr/bin/env python3
"""function def learning_rate_decay(alpha, decay_rate,global_step, decay_step):
that creates a learning rate decay operation in tensorflow
using inverse time decay:"""
import tensorflow as tf


def batch_norm(Z, gamma, beta, epsilon):
    """creates a learning rate decay operation in tensorflow
    using inverse time decay"""
    t = tf.compat.train.inverse_time_decay(
        alpha,
        global_step,
        decay_steps,
        decay_rate,
        staircase=True)
    return t
