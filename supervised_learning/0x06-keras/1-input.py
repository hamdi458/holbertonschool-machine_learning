#!/usr/bin/env python3
"""function def build_model(nx, layers, activations, lambtha, keep_prob):
that builds a neural network with the Keras library:"""

import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    x = k.Input(shape=(nx,))
    for i in range(len(layers)):
        if i == 0:
            y = (k.layers.Dense(layers[i], activation=activations[i],
                 kernel_regularizer=k.regularizers.l2(lambtha)))(x)
        else:
            y = (k.layers.Dense(layers[i], activation=activations[i],
                 kernel_regularizer=k.regularizers.l2(lambtha)))(y)
        if i < len(layers) - 1:
            y = (k.layers.Dropout(1-keep_prob))(y)
    model = k.Model(x, y)
    return model
