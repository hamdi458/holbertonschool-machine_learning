#!/usr/bin/env python3
""" creates the training operation for a neural network in tensorflow
using the gradient descent with momentum optimization algorithm """
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ creates the training operation for a neural network in tensorflow
    using the gradient descent with momentum optimization algorithm"""
    op_n_n = tf.train.MomentumOptimizer(alpha, beta1)
    mo_op_op = op_n_n.minimize(loss)
    return mo_op_op
