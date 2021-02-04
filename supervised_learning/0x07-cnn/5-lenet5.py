#!/usr/bin/env python3
"""builds a modified version of the LeNet-5 architecture using keras"""
import tensorflow.keras as K


def lenet5(X):
    """that builds a modified version of the LeNet5 architecture using keras"""
    kernel = K.initializers.he_normal(seed=None)
    lay1 = K.layers.Conv2D(kernel_size=(5, 5),
                           strides=1, padding='SAME',
                           kernel_initializer=kernel,
                           filters=6, activation='relu')(X)
    lay2 = K.layers.MaxPooling2D(
                                  strides=2,
                                  pool_size=2)(lay1)
    lay3 = K.layers.Conv2D(
                            filters=16,
                            kernel_size=5,
                            padding='VALID',
                            activation='relu',
                            kernel_initializer=kernel)(lay2)
    lay4 = K.layers.MaxPooling2D(
                                  strides=(2, 2),
                                  pool_size=(2, 2))(lay3)
    flat = K.layers.Flatten(data_format=None)(lay4)

    lay5 = K.layers.Dense(
                           units=120,
                           kernel_initializer=kernel,
                           activation='relu')(flat)
    lay6 = K.layers.Dense(
                           units=84,
                           kernel_initializer=kernel,
                           activation='relu')(lay5)
    lay7 = K.layers.Dense(activation='softmax',
                          units=10,
                          kernel_initializer=kernel)(lay6)
    network = K.Model(inputs=X, outputs=lay7)
    opt = K.optimizers.Adam()
    network.compile(loss='categorical_crossentropy', optimizer=opt,
                    metrics=['accuracy'])
    return network
