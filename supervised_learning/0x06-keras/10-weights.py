#!/usr/bin/env python3
"""save and load model"""

import tensorflow.keras as K


def save_weights(network, filename):
    """saves an weights"""
    network.save_weights(filename)
    return None


def load_weights(filename):
    """loads an weights"""
    network.load_weights(filename)
    return None
