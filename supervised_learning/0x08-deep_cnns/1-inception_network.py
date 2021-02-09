#!/usr/bin/env python3
"""inception network"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """inception network"""
    input_layer = K.Input(shape=(224, 224, 3))
    kernel_init = K.initializers.he_normal(seed=None)
    x = K.layers.Conv2D(64, kernel_size=(7, 7), padding='same', strides=(2, 2),
                        activation='relu',
                        kernel_initializer=kernel_init)(input_layer)
    x = K.layers.MaxPool2D(pool_size=(3, 3), padding='same', strides=(2, 2))(x)
    x = K.layers.Conv2D(64, kernel_size=(1, 1),
                        activation='relu',
                        kernel_initializer=kernel_init)(x)

    x = K.layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1),
                        activation='relu',
                        kernel_initializer=kernel_init)(x)

    x = K.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_block(x, [64, 96, 128, 16, 32, 32])

    x = inception_block(x, [128, 128, 192, 32, 96, 64])

    x = K.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_block(x, [192, 96, 208, 16, 48, 64])
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])

    x = K.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    x = K.layers.AveragePooling2D((7, 7), strides=1)(x)
    x = K.layers.Dropout(0.4)(x)
    x = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=kernel_init)(x)

    output = K.Model(inputs=input_layer, outputs=x)
    return output
