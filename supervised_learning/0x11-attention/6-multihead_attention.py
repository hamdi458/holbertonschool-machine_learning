#!/usr/bin/env python3
""" Multi Head Attention """
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ Perform multi head attention """

    def __init__(self, dm, h):
        """
        Method:
        -------
            Class constructor
        Parameters:
        -----------
            dm (integer): representing the dimensionality of the model.
            h (integer): representing the number of heads.
        **
        dm is divisible by h
        **
        """
        # Sets the following public instance attributes:

        #     h - the number of heads
        #     dm - the dimensionality of the model
        #     depth - the depth of each attention head
        #     Wq - a Dense layer with dm units,
        #       used to generate the query matrix
        #     Wk - a Dense layer with dm units,
        #       used to generate the key matrix
        #     Wv - a Dense layer with dm units,
        #       used to generate the value matrix
        #     linear - a Dense layer with dm units,
        #       used to generate the attention output
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """
        Public Instance Method
        Q: tensor shape (batch, seq_len_q, dk) contains input to
            generate the query matrix
        K: tensor shape (batch, seq_len_v, dk) contains input to
            generate the key matrix
        V: tensor shape (batch, seq_len_v, dv) contains input to
            generate the value matrix
        Returns:
            output: tensor with last two dims (..., seq_len_q, dm)
                contains scaled dot product attention
            w: tensor with last three dims
                (..., h, seq_len_q, seq_len_v) contains attention w
        """
        batch_size = tf.shape(Q)[0]
        attention_parameters = [
            self.Wq(Q),
            self.Wk(K),
            self.Wv(V)
        ]
        for i, parameter in enumerate(attention_parameters):
            # Split the feature axis into heads x depth, where depth is a
            # subset/slice of the features
            # Then, swap the heads & tokens axes
            attention_parameters[i] = tf.transpose(
                tf.reshape(
                    parameter, (*parameter.shape[:-1], self.h, self.depth)
                ),
                perm=[0, 2, 1, 3]
            )
        softmax, output1 = sdp_attention(attention_parameters, mask)
        softmax = tf.transpose(softmax, perm=[0, 2, 1, 3])
        concat = tf.reshape(softmax, (batch_size, -1, self.dm))
        output = self.linear(concat)
        return output, output1