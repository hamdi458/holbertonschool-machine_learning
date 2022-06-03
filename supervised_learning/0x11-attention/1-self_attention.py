#!/usr/bin/env python3
"""
calculate the attention for machine translation
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """calculate the attention for machine translation"""
    def __init__(self, units):
        """
        ARGS:
            *units :{integer}:  the number of hidden units in
                the alignment model
        Sets :
            *W - {Dense} units units: the previous decoder hidden state
            *U - {Dense} units units: the encoder hidden states
            *V - {Dense} 1 units: the tanh of the sum of the outputs
                of W and U
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        args:
            *s_prev is a tensor of shape (batch, units)
                containing the previous decoder hidden state
            *hidden_states is a tensor of shape
                (batch, input_seq_len, units):  the outputs of the encoder
        Returns: context, weights
            *context is a tensor of shape (batch, units)
                that contains the context vector for the decoder
            *weights is a tensor of shape (batch, input_seq_len, 1)
                that contains the attention weights
        """

        query = s_prev
        values = hidden_states

        newaxis_query = tf.expand_dims(query, 1)
        a = self.W(newaxis_query)

        b = self.U(values)
        # Calculate attention scores for Input
        score = self.V(tf.nn.tanh(a + b))

        # Calculate softmax
        score = tf.nn.softmax(score, axis=1)

        # Multiply scores with values
        weighted_values = score * values
        # Sum weighted values to get Output 1
        context = tf.reduce_sum(weighted_values, axis=1)
        return context, score
