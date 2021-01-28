#!/usr/bin/env python3
"""that performs a convolution on grayscale images:"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """that performs a convolution on grayscale images:"""
    m = images.shape[0]
    sh, sw = stride
    input_w, input_h = images.shape[2], images.shape[1]
    filter_w, filter_h = kernel.shape[1], kernel.shape[0]
    if padding == 'valid':
        output_h = int(((input_h - filter_h + 1) / float(stride[0])))
        output_w = int(((input_w - filter_w + 1) / float(stride[1])))
        output = np.zeros((m, output_h, output_w))

        for x in range(output_w):
            for y in range(output_h):
                output[:, y, x] = np.sum(kernel * images[:,
                                         y * stride[0]:y * stride[0]+filter_h,
                                   x * stride[1]:x *
                                   stride[1] + filter_w], axis=(1, 2))
        return output

    elif padding == 'same':
        ph = ((input_h - 1) * sh + filter_h - input_h) / 2 + 1
        pw = ((input_w - 1) * sw + filter_w - input_w) / 2 + 1
        output = np.zeros((m, input_h, input_w))
        image_padded = np.pad(images, ((0,), (int(ph),), (int(pw),)),)
        for x in range(input_w):
            for y in range(input_h):
                output[:, y, x] = np.sum(
                    image_padded[:, y * sh: y * sh + filter_h,
                                 x * sw: x * sw + filter_w] * kernel,
                    axis=(1, 2))
        return output
    else:
        m, h, w = images.shape
        kh, kw = kernel.shape
        ph, pw = padding
        filter_w, filter_h = kernel.shape[1], kernel.shape[0]
        oh = h - kh + (2 * ph) + 1
        output_h = (h - kh + (2 * ph) + 1) // sh
        output_w = (w - kw + (2 * pw) + 1) // sw
        image_padded = np.zeros((m, h + oh, w + w - kw + (2 * pw) + 1))
        image_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),)
        output = np.zeros((m, output_h, output_w))
        for x in range(output_w):
            for y in range(output_h):
                output[:, y, x] = np.sum(image_padded[
                    :, y * sh: y * sh + filter_h,
                    x * sw: x * sw + filter_w] * kernel,
                    axis=(1, 2))
        return output
