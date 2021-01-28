#!/usr/bin/env python3
""" that performs a convolution on images with channels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """convolution with channels"""
    m = images.shape[0]
    input_w, input_h = images.shape[2], images.shape[1]
    filter_w, filter_h = kernels.shape[1], kernels.shape[0]
    output_d = kernels.shape[3]
    sh, sw = stride

    if padding == 'valid':
        output_h = int((input_h - filter_h) // sh + 1)
        output_w = int((input_w - filter_w) // sw + 1)
        output = np.zeros((m, output_h, output_w, output_d))
        for ch in range(output_d):
            for x in range(output_w):
                for y in range(output_h):
                    output[:, y, x, ch] = (kernels[:, :, :, ch] *
                                           images[:,
                                           y*stride[0]:y*stride[0]+filter_h,
                                           x*stride[1]:x*stride[1]+filter_w,
                                           :]).sum(axis=(1, 2, 3))
        return output
    elif padding == 'same':
        ph = ((input_h - 1) * sh + filter_h - input_h) / 2 + 1
        pw = ((input_w - 1) * sw + filter_w - input_w) / 2 + 1
        ph, pw = int(ph), int(pw)
        output = np.zeros((m, input_h, input_w, output_d))
        image_padded = np.zeros((m, int(input_h + 2 * ph),
                                 int(input_w + 2 * pw), images.shape[3]))
        image_padded[:, ph:input_h + ph, pw:input_w + pw, :] = images
        output_h = int((input_h - filter_h) // sh + 1)
        output_w = int((input_w - filter_w) // sw + 1)

        for ch in range(output_d):
            for x in range(output_w):
                for y in range(output_h):
                    output[:, y, x, ch] = (kernels[:, :, :, ch] *
                                           images[:,
                                           y*stride[0]:y*stride[0]+filter_h,
                                           x*stride[1]:x*stride[1]+filter_w,
                                           :]).sum(axis=(1, 2, 3))
        return output
    else:
        ph, pw = padding

        image_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

        output_h = (input_h - filter_h + (2 * ph)) // sh + 1
        output_w = (input_w - filter_w + (2 * pw)) // sw + 1

        output = np.zeros((m, int(output_h), int(output_w)))
        for x in range(output_w):
            for y in range(output_h):
                output[:, y, x] = np.sum(
                    image_padded[:, y * sh: y * sh + filter_h,
                                 x * sw: x * sw + filter_w, :] * kernel,
                    axis=(1, 2, 3))
        return output
