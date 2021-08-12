#!/usr/bin/env python3
"""3. Shear"""
import tensorflow as tf


def shear_image(image, intensity):
    """
     function def shear_image(image, intensity):
        that randomly shears an image:
    """
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.preprocessing.image.random_shear(img, intensity,
                                                        row_axis=0,
                                                        col_axis=1,
                                                        channel_axis=2)
    image = tf.keras.preprocessing.image.array_to_img(image)
    return image
