#!/usr/bin/env python3
"""
based on:
https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
Multihead Attention
"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi Head Attention Class
    """
    def __init__(self, dm, h):
        """ Function that initializes """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split(self, x, batch_size):
        """ Function that """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """ Function that returns output, weights """
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        q = self.split(q, batch_size)
        k = self.split(k, batch_size)
        v = self.split(v, batch_size)
        sc_attention, attention_weights = sdp_attention(q, k, v, mask)
        sc_attention = tf.transpose(sc_attention,
                                    perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(sc_attention,
                                      (batch_size, -1, self.dm))
        output = self.linear(concat_attention)
        return output, attention_weights
