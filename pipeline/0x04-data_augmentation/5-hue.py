#!/usr/bin/env python3
"""5.Hue"""
import tensorflow as tf


def change_hue(image, delta):
    """function def change_hue(image, delta):
        that changes the hue of an image:"""

    image = tf.image.adjust_hue(image, delta)

    return image
