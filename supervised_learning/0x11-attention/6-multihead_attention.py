#!/usr/bin/env python3
""" Multi Head Attention"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention class"""
    def __init__(self, dm, h):
        """initialize class constructor"""
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm

        assert dm % self.h == 0

        self.depth = dm // self.h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (h, depth).
        Transpose the result such that the shape is(batch_size,h,seq_len,depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """perform multi head attention"""
        batch_size = tf.shape(Q)[0]
        # (batch_size, seq_len, dm)
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        # (batch_size, h, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, h, seq_len_q, depth)
        # attention_weights.shape == (batch_size, h, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        output = self.linear(concat_attention)

        return output, attention_weights
