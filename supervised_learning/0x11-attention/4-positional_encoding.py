#!/usr/bin/env python3
""" Positional Encoding"""
import numpy as np




def positional_encoding(max_seq_len, dm):
    """ calculates the positional encoding for a transformer"""
    def calangels(position, i, d_model):
        rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return position * rates
    angle_rads = calangels(np.arange(max_seq_len)[:, np.newaxis],
                            np.arange(dm)[np.newaxis, :], dm)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads