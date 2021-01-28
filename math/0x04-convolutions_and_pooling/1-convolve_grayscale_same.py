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
    m, h, w = images.shape
    mc = 0
    Kr, Kc = kernel.shape
    kr = Kr // 2
    kc = Kc // 2
    for dr in range(-kr, kr+1, 1):
        mr = np.roll(images, -dr, axis=1)
        for dc in range(-kc, kc + 1, 1):
            mrc = np.roll(mr, -dc, axis=2)
            mc = mc+kernel[dr+kr, dc+kc]*mrc
    return mc
