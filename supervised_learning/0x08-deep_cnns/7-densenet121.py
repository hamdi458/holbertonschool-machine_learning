#!/usr/bin/env python3
"""DenseNet-121 builds the DenseNet-121 architecture"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    growth_rate is the growth rate
    compression is the compression factor
    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization and
        a rectified linear activation (ReLU), respectively
    All weights should use he normal initialization
    You may use:
        dense_block = __import__('5-dense_block').dense_block
        transition_layer = __import__('6-transition_layer').transition_layer
    Returns: the keras model
    """
    kernel = K.initializers.he_normal(seed=None)
    img_input = K.Input(shape=[224, 224, 3])
    X = K.layers.BatchNormalization()(img_input)
    X = K.layers.ReLU()(X)
    filters = 0
    if filters <= 0:
        filters = 2 * growth_rate
    X = K.layers.Conv2D(filters=filters,
                        kernel_size=(7, 7), strides=2, padding="same",
                        kernel_initializer=kernel)(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(X)
    X, filters = dense_block(X, filters, growth_rate, 6)
    X, filters = transition_layer(X, filters, compression)
    X, filters = dense_block(X, filters, growth_rate, 12)
    X, filters = transition_layer(X, filters, compression)
    X, filters = dense_block(X, filters, growth_rate, 24)
    X, filters = transition_layer(X, filters, compression)
    X, filters = dense_block(X, filters, growth_rate, 16)
    X = K.layers.AveragePooling2D(pool_size=7)(X)
    output = K.layers.Dense(units=1000, activation='softmax',
                            kernel_initializer=kernel)(X)
    return K.models.Model(inputs=img_input, outputs=output)
