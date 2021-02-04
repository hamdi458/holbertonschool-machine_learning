#!/usr/bin/env python3
"""builds a modified version of the LeNet-5 architecture using tensorflow"""
import tensorflow as tf


def lenet5(x, y):
    """builds a modified version of the LeNet5 architecture using tensorflow"""
    kernel = tf.contrib.layers.variance_scaling_initializer()
    lay1 = tf.layers.conv2d(x, kernel_size=(5, 5),
                            strides=1, padding='SAME',
                            kernel_initializer=kernel,
                            filters=6, activation='relu')
    lay2 = tf.layers.MaxPooling2D(
                                  strides=(2, 2),
                                  pool_size=[2, 2])(lay1)
    lay3 = tf.layers.Conv2D(
                            filters=16,
                            kernel_size=5,
                            padding='VALID',
                            activation='relu',
                            kernel_initializer=kernel)(lay2)
    lay4 = tf.layers.MaxPooling2D(
                                  strides=(2, 2),
                                  pool_size=(2, 2))(lay3)
    fc0 = tf.contrib.layers.flatten(lay4)

    lay5 = tf.layers.Dense(
                           units=120,
                           kernel_initializer=kernel,
                           activation='relu')(fc0)
    lay6 = tf.layers.Dense(
                           units=84,
                           kernel_initializer=kernel,
                           activation='relu')(lay5)
    lay7 = tf.layers.Dense(
                           units=10,
                           kernel_initializer=kernel,
                           activation='softmax')(lay6)
    softmax = tf.nn.softmax(lay7)
    pred = tf.equal(tf.argmax(y, axis=1), tf.argmax(lay7, axis=1))
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))
    loss = tf.losses.softmax_cross_entropy(y, lay7)

    opt = tf.train.AdamOptimizer().minimize(loss)

    return softmax, opt, loss, acc
