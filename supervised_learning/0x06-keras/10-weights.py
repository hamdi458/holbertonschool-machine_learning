#!/usr/bin/env python3
"""save and load model"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """saves an weights"""
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """loads an weights"""
    network.load_weights(filename)
    return None
