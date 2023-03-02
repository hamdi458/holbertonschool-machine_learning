#!/usr/bin/env python3
""" Tranformer decoder block"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ class DecoderBlock """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ Function that initializes """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """ Function that returns a tensor containing the block’s output """
        att, att_b = self.mha1(x, x, x, look_ahead_mask)
        att = self.dropout1(att, training=training)
        outt = self.layernorm1(att + x)
        atten, attn_weights_block2 = self.mha2(outt, encoder_output,
                                               encoder_output,
                                               padding_mask)
        atten = self.dropout2(atten, training=training)
        outtn = self.layernorm2(atten + outt)
        hidden_outp = self.dense_hidden(outtn)
        outpp = self.dense_output(hidden_outp)
        outpn = self.dropout3(outpp, training=training)
        outp = self.layernorm3(outpn + outtn)
        return outp
