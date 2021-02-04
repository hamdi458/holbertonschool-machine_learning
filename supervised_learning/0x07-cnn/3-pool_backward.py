#!/usr/bin/env python3
"""back propagation over a pooling layer of a neural network"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """back propagation over a pooling layer of a neural network"""

    (m, h_new, w_new, c_new) = dA.shape
    (m, h_prev, w_prev, c) = A_prev.shape
    (kh, kw) = kernel_shape
    (sh, sw) = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):  # loop on the horizontal axis
                for c in range(c_new):  # loop over the channels (depth)
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        # Use the corners and "c" to define the current slice
                        # from a_prev
                        a_prev_slice = a_prev[vert_start:vert_end,
                                              horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)

                        mask = a_prev_slice == np.max(a_prev_slice)
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end,
                                c] += np.multiply(mask, dA[i, h, w, c])

                    elif mode == "average":
                        print('w')
                        # Get the value a from dA (≈1 line)
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        average = dz / (kh * kw)
                        a = np.ones(kh, kw) * average
                        # Distribute it to get the correct slice of dA_prev
                        # Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] += a
    return dA_prev
