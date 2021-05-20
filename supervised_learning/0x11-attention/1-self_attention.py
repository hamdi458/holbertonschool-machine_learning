#!/usr/bin/env python3
""" class RNNCell """
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ the class SelfAttention """
    def __init__(self, units):
        """initializer Class constructor"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """calculate the attention for machine translation
            based on this paper:"""
        query_with_time_axis = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) +
                       self.W2(hidden_states)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
