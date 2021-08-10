#!/usr/bin/env python3
"""0. Flip"""
import tensorflow as tf


def flip_image(image):
    """function def flip_image(image): that flips an image horizontally"""

    image = tf.image.flip_left_right(image)

    return image
