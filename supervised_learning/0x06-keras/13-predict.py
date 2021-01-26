#!/usr/bin/env python3
"""makes a prediction using a neural network"""
import tensorflow.keras as k


def predict(network, data, verbose=False):
    """makes a prediction using a neural network"""
    predic = network.predict(x=data, verbose=verbose)
    return predic
