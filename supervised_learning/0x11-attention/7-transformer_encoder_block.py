#!/usr/bin/env python3
""" Transformer Encoder Block """

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """ Encoder Block class"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ Function that initilizes """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """ Function that returns a tensor that contains block’s output """
        att_outp, _ = self.mha(x, x, x, mask)
        att_outp = self.dropout1(att_outp, training=training)
        outt = self.layernorm1(x + att_outp)
        h_outp = self.dense_hidden(outt)
        h_outp = self.dense_output(h_outp)
        h_outp = self.dropout2(h_outp, training=training)
        outp = self.layernorm2(outt + h_outp)
        return outp
