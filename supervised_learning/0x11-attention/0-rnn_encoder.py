#!/usr/bin/env python3
""" the class RNNCell """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """  the class RNNCell """
    def __init__(self, vocab, embedding, units, batch):
        """initializer Class constructor"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer="glorot_uniform")

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNN cell
        to a tensor of zeros"""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Returns: outputs, hidden
        outputs is a tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder
        hidden is a tensor of shape (batch, units)
            containing the last hidden state of the encoder"""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
