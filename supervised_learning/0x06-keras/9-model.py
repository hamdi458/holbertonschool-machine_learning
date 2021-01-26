#!/usr/bin/env python3
"""save and load model"""

import tensorflow.keras as K


def save_model(network, filename):
    """saves an entire"""
    network.save(filename)
    return None


def load_model(filename):
    """loads an entire"""
    loads = K.models.load_model(filename)
    return loads
