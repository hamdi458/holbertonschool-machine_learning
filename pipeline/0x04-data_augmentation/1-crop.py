#!/usr/bin/env python3
"""1. Crop"""
import tensorflow as tf


def crop_image(image, size):
    """function def crop_image(image, size):
        that performs a random crop of an image"""

    image = tf.image.random_crop(image, size, seed=None, name=None)

    return image
