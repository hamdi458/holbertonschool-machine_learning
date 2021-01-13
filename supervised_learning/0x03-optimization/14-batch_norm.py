#!/usr/bin/env python3
"""function def create_batch_norm_layer(prev, n, activation):
that creates a batch normalization layer
for a neural network in tensorflow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    '''batch normalization layer for a neural network in tensorflow'''
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    inputs = tf.layers.dense(
         prev, n, activation=None, use_bias=True, kernel_initializer=kernel,
         bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
         bias_regularizer=None, activity_regularizer=None,
         kernel_constraint=None,
         bias_constraint=None, trainable=True, name='layer', reuse=None)
    beta = tf.Variable(tf.zeros([n]))
    gamma = tf.Variable(tf.ones([n]))
    mean, var = tf.nn.moments(inputs, axes=[0])
    la = tf.nn.batch_normalization(
         inputs, mean, var, beta, gamma, 1e-8, name=None)
    return activation(la)
