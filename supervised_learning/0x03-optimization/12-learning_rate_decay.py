#!/usr/bin/env python3
"""function def learning_rate_decay(alpha, decay_rate,global_step, decay_step):
that creates a learning rate decay operation in tensorflow
using inverse time decay:"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """creates a learning rate decay operation in tensorflow
    using inverse time decay"""
    t = tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True)
    return t
