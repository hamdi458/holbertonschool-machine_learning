#!/usr/bin/env python3
"""Transition Layer
builds a transition layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer
    Your code should implement compression as used in DenseNet-C
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization and
        a rectified linear activation (ReLU), respectively
    Returns: The output of the transition layer and the number of
        filters within the output, respectively"""
    kernel = K.initializers.he_normal(seed=None)
    filters = nb_filters * compression
    output = K.layers.BatchNormalization()(X)
    output = K.layers.ReLU()(output)
    output = K.layers.Conv2D(int(filters),
                             (1, 1), padding="same",
                             kernel_initializer=kernel)(output)
    output = K.layers.AveragePooling2D(pool_size=(2, 2),
                                       strides=(2, 2))(output)
    return output, int(filters)
