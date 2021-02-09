#!/usr/bin/env python3
"""inception block keras"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """inception blok keras"""
    F1, F3R, F3, F5R, F5, FPP = filters
    kernel = K.initializers.he_normal(seed=None)
    conv1 = K.layers.Conv2D(kernel_size=(1, 1),
                            strides=1, padding='same',
                            kernel_initializer=kernel,
                            filters=F1, activation='relu')(A_prev)
    conv2_1 = K.layers.Conv2D(kernel_size=(1, 1),
                              strides=1, padding='same',
                              kernel_initializer=kernel,
                              filters=F3R, activation='relu')(A_prev)
    conv2_2 = K.layers.Conv2D(kernel_size=(3, 3),
                              strides=1, padding='same',
                              kernel_initializer=kernel,
                              filters=F3, activation='relu')(conv2_1)
    conv3_1 = K.layers.Conv2D(kernel_size=(1, 1),
                              strides=1, padding='same',
                              kernel_initializer=kernel,
                              filters=F5R, activation='relu')(A_prev)
    conv3_2 = K.layers.Conv2D(kernel_size=(5, 5),
                              strides=1, padding='same',
                              kernel_initializer=kernel,
                              filters=F5, activation='relu')(conv3_1)
    pool4_1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                    padding='same')(A_prev)
    conv4_2 = K.layers.Conv2D(kernel_size=(1, 1),
                              strides=1, padding='same',
                              kernel_initializer=kernel,
                              filters=FPP, activation='relu')(pool4_1)
    filter_concat = K.layers.concatenate([conv1, conv2_2, conv3_2, conv4_2])
    return filter_concat
