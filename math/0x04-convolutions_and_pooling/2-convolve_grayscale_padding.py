#!/usr/bin/env python3
"""function def convolve_grayscale_same(images, kernel):
that performs a same convolution on grayscale images:"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    filter_w, filter_h = kernel.shape[1], kernel.shape[0]
    oh = h - kh + (2 * ph) + 1
    padded_images = np.zeros((m, h + oh, w + w - kw + (2 * pw) + 1))
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),)
    output = np.zeros((m, h - kh + (2 * ph) + 1, w - kw + (2 * pw) + 1))
    for x in range(w - kw + (2 * pw) + 1):
        for y in range(h - kh + (2 * ph) + 1):
            output[:, y, x] = (kernel * padded_images[
                :, y: y + kh, x: x + kw]).sum(axis=(1, 2))
    return output
