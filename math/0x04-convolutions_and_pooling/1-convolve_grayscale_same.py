#!/usr/bin/env python3
"""function def convolve_grayscale_same(images, kernel)"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
        for the convolution
            kh is the height of the kernel
            kw is the width of the kernel
            Returns: a numpy.ndarray containing the convolved images"""
    m = images.shape[0]
    input_h, input_w = images.shape[1], images.shape[2]
    filter_w, filter_h = kernel.shape[1], kernel.shape[0]
    padded_images = np.pad(images, ((0,), (filter_h//2,), (filter_w//2,)),)
    output = np.zeros((m, input_h, input_w))
    for x in range(input_w):
        for y in range(input_h):
            output[:, y, x] = (kernel *
                               padded_images[:, y:y+filter_h,
                                      x:x+filter_w]).sum(axis=(1, 2))
    return output
