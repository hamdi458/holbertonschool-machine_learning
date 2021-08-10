#!/usr/bin/env python3
"""2. Rotate"""
import tensorflow as tf


def rotate_image(image):
    """function def rotate_image(image):
        that rotates an image by 90 degrees counter-clockwise:"""
    image = tf.image.rot90(image, k=1)
    return image
