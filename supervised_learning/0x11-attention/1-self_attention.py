#!/usr/bin/env python3
""" self attention class """
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ Self attention class """
    def __init__(self, units):
        """ initialization function """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """ Function that returns context and weights"""
        ss_prev = tf.expand_dims(s_prev, axis=1)
        score_t = self.V(tf.nn.tanh(self.W(ss_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score_t, axis=1)
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)
        return context, weights
