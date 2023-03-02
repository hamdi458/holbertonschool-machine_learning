#!/usr/bin/env python3
""" postitional encodingn """

import numpy as np


def positional_encoding(max_seq_len, dm):
    """ Function that calculates the positional
        encoding for a transformer"""
    enc = np.arange(max_seq_len)[:, np.newaxis]
    x = np.arange(dm)[np.newaxis, :]
    flt = np.float32(dm)
    grad_angle = 1 / (np.power(10000, (2 * (x // 2) / flt)))
    angle = enc * grad_angle
    pos = np.zeros((max_seq_len, dm))
    pos[:, 0::2] = np.sin(angle[:, 0::2])
    pos[:, 1::2] = np.cos(angle[:, 1::2])
    return pos
