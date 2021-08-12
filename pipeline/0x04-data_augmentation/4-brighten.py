#!/usr/bin/env python3
"""4. Brightness"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    function def change_brightness(image, max_delta):
        that randomly changes the brightness of an image:
    """
    image = tf.image.adjust_brightness(image, max_delta)
    return image
