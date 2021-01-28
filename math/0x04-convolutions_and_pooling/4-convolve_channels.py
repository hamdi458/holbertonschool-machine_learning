#!/usr/bin/env python3
""" that performs a convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """convolution with channels"""
    m = images.shape[0]
    input_w, input_h = images.shape[2], images.shape[1]
    filter_w, filter_h = kernel.shape[1], kernel.shape[0]
    output_d = kernel.shape[2]
    sh, sw = stride

    if padding == 'valid':

        output_h = int((input_h - filter_h) // sh + 1)
        output_w = int((input_w - filter_w) // sw + 1)

        output = np.zeros((m, output_h, output_w))
        for x in range(output_w):
            for y in range(output_h):
                output[:, y, x] = (kernel *
                                   images[:,
                                          y * stride[0]:y * stride[0]+filter_h,
                                          x * stride[1]:x * stride[1]+filter_w,
                                          :]).sum(axis=(1, 2, 3))
        return output
