#!/usr/bin/env python3
""" saves and loads a model’s weights:"""
import tensorflow.keras as K


def save_config(network, filename):
    """ saves a model’s weights:"""
    with open(filename, "w") as fi:
        fi.write(network.to_json())
    return None


def load_config(filename):
    """loads a model’s weights"""
    with open(filename, "r") as fi:
        r = fi.read()
    load = k.models.model_from_json(r)
    return load
