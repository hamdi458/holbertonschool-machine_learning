#!/usr/bin/env python3
"""function def convolve_grayscale_valid(images, kernel)"""
import numpy as np
from math import ceil, floor


def convolve_grayscale_valid(images, kernel):
    """performs a valid convolution on grayscale images:
    images is a numpy.ndarray with shape (m, h, w) containing multiple
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
    output_h = int(ceil(float(input_h - filter_h + 1)))
    output_w = int(ceil(float(input_w - filter_w + 1)))
    output = np.zeros((m, output_h, output_w))
    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] = (kernel *
                               images[:, y:y+filter_h,
                                      x:x+filter_w]).sum(axis=(1, 2))
    return output