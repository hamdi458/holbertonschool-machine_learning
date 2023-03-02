#!/usr/bin/env python3
"""Transformer Encoder"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Encoder class"""
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """initialization Class constructor"""
        super(Encoder, self).__init__()
        self.dm = dm
        self.N = N
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """ Function that returns tensor containing
            the encoder output """
        seq_len = x.shape[1]
        emb = self.embedding(x)
        emb *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        emb += self.positional_encoding[:seq_len]
        enc_outp = self.dropout(emb, training=training)
        for i in range(self.N):
            enc_outp = self.blocks[i](enc_outp, training, mask)
        return enc_outp
