#!/usr/bin/env python3
"""Write a function def test_model(network, data, labels, verbose=True):
that tests a neural network
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """tests a neural network"""
    evaluat = network.evaluate(x=data, y=labels, verbose=verbose)
    return evaluat
