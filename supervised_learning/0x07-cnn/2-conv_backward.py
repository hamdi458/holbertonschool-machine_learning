#!/usr/bin/env python3

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """"""

    (kh, kw, c_prev, c_new) = W.shape
    m, h_new, w_new, c_new = dZ.shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    sh, sw = stride
    print(dZ.shape)
    print(A_prev.shape)

    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.zeros((1, 1, 1, c_new))

    dA_prev_pad = np.zeros((m, h_prev, w_prev, c_prev))
    if padding == 'valid':
        for i in range(m):

            a_prev_pad = A_prev[i]
            da_prev_pad = dA_prev_pad[i]

            for h in range(h_new):
                for w in range(w_new):
                    for c in range(c_new):
                        vert_start = h
                        vert_end = vert_start + kh
                        horiz_start = w
                        horiz_end = horiz_start + kw

                        a_slice = a_prev_pad[vert_start*sh:vert_end,
                                             horiz_start*sw:horiz_end, :]

                        da_prev_pad[vert_start*sh:vert_end,
                                    horiz_start*sw:horiz_end,
                                    :] += W[:, :, :, c] * dZ[i, h, w, c]
                        dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]

            dA_prev[i, :, :, :] = da_prev_pad[0:h_prev, 0: w_prev, :]

        return dA_prev, dW, db
    elif padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - kh % 2 - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - kw % 2 - w_prev) / 2) + 1
        A_prev_pad = np.pad(A_prev, pad_width=((0, 0),
                                               (ph, ph), (pw, pw), (0, 0)))
        dA_prev_pad = np.pad(dA_prev,
                             pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)))
        for i in range(m):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            for h in range(h_new):
                for w in range(w_new):
                    for c in range(c_new):

                        vert_start = h * sh
                        vert_end = vert_start + kh
                        horiz_start = w * sw
                        horiz_end = horiz_start + kw

                        a_slice = a_prev_pad[vert_start:vert_end,
                                             horiz_start:horiz_end, :]

                        da_prev_pad[vert_start:vert_end,
                                    horiz_start:horiz_end, :] += W[
                                        :, :, :, c] * dZ[i, h, w, c]
                        dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]
                dA_prev[i, :, :, :] = da_prev_pad[ph:h_prev + ph,
                                                  pw: w_prev + pw, :]
        return da_prev_pad, dW, db
    else:
        ph, pw = padding
        A_prev_pad = np.pad(A_prev, pad_width=((0, 0),
                                               (ph, ph), (pw, pw), (0, 0)))
        dA_prev_pad = np.pad(dA_prev,
                             pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)))
        for i in range(m):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            for h in range(h_new):
                for w in range(w_new):
                    for c in range(c_new):

                        vert_start = h * sh
                        vert_end = vert_start + kh
                        horiz_start = w * sw
                        horiz_end = horiz_start + kw

                        a_slice = a_prev_pad[vert_start:vert_end,
                                             horiz_start:horiz_end, :]

                        da_prev_pad[vert_start:vert_end,
                                    horiz_start:horiz_end, :] += W[
                                        :, :, :, c] * dZ[i, h, w, c]
                        dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]
                dA_prev[i, :, :, :] = da_prev_pad[ph:h_prev + ph,
                                                  pw: w_prev + pw, :]
        return da_prev_pad, dW, db
