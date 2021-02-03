#!/usr/bin/env python3
"""performs pooling on images:"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs pooling on images:"""
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = int((h - kh) / sh + 1)
    output_w = int((w - kw) / sw + 1)

    output = np.zeros((m, output_h, output_w, c))
    for x in range(output_h):
        for y in range(output_w):
            if mode == 'max':
                output[:, x, y, :] = np.max(
                    A_prev[:, x * sh: x * sh + kh, y * sw: y * sw + kw, :],
                    axis=(1, 2))
            else:
                output[:, x, y, :] = np.mean(
                    A_prev[:, x * sh: x * sh + kh, y * sw: y * sw + kw, :],
                    axis=(1, 2))
    return output
