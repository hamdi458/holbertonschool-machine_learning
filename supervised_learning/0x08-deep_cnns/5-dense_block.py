#!/usr/bin/env python3
"""dence block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ X is the output from the previous layer
        nb_filters is an integer representing the number of filters in X
        growth_rate is the growth rate for the dense block
        layers is the number of layers in the dense block
        You should use the bottleneck layers used for DenseNet-B
        All weights should use he normal initialization
        All convolutions should be preceded by Batch Normalization and
            a rectified linear activation (ReLU), respectively
        Returns: The concatenated output of each layer within the
            Dense Block and the number of filters within the
            concatenated outputs, respectively"""
    output = X
    kernel = K.initializers.he_normal(seed=None)

    for i in range(layers):
        dense_factor = K.layers.BatchNormalization()(output)
        dense_factor = K.layers.ReLU()(dense_factor)
        interchannel = growth_rate * 4
        dense_factor = K.layers.Conv2D(interchannel,
                                       (1, 1),
                                       kernel_initializer=kernel,
                                       padding='same')(dense_factor)
        """bottleneck finish"""                               
        dense_factor = K.layers.BatchNormalization()(dense_factor)
        dense_factor = K.layers.ReLU()(dense_factor)
        dense_factor = K.layers.Conv2D(growth_rate, (3, 3),
                                       kernel_initializer=kernel,
                                       padding='same')(dense_factor)
        output = K.layers.concatenate([output, dense_factor])
        nb_filters += growth_rate
    return output, nb_filters
