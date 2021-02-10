#!/usr/bin/env python3
"""projection Block """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12  is the number of filters in the second 1x1 convolution as well
            as the 1x1 convolution in the shortcut connection
    s is the stride of the first convolution in both the main path and
        the shortcut connection
    All convolutions inside the block should be followed by batch
        normalization along the channels axis and a rectified linear
        activation (ReLU), respectively.
    All weights should use he normal initialization
    Returns: the activated output of the projection block"""
    F11, F3, F12 = filters
    kernel = K.initializers.he_normal(seed=None)
    conv1 = K.layers.Conv2D(kernel_size=(1, 1),
                            strides=s, padding='same',
                            kernel_initializer=kernel,
                            filters=F11)(A_prev)
    conv1 = K.layers.BatchNormalization()(conv1)
    conv1 = K.layers.Activation('relu')(conv1)
    conv2 = K.layers.Conv2D(kernel_size=(3, 3),
                            strides=1, padding='same',
                            kernel_initializer=kernel,
                            filters=F3)(conv1)
    conv2 = K.layers.BatchNormalization()(conv2)
    conv2 = K.layers.Activation('relu')(conv2)
    conv3 = K.layers.Conv2D(kernel_size=(1, 1),
                            strides=1, padding='same',
                            kernel_initializer=kernel,
                            filters=F12,)(conv2)
    conv3 = K.layers.BatchNormalization()(conv3)

    conv_proj = K.layers.Conv2D(kernel_size=(1, 1),
                                strides=s, padding='same',
                                kernel_initializer=kernel,
                                filters=F12,)(A_prev)
    conv_proj = K.layers.BatchNormalization()(conv_proj)
    al = K.layers.Add()([conv3, conv_proj])

    output = K.layers.Activation('relu')(al)
    return output
