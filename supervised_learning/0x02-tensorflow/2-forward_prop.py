#!/usr/bin/env python3
"""forward propagation graph for the neural network"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """forward propagation graph for the neural network"""
    for i in range(len(layer_sizes)):
        layer = create_layer(x, layer_sizes[i], activations[i])
        x = layer
    return layer
