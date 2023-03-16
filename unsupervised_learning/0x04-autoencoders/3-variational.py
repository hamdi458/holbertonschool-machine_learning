#!/usr/bin/env python3

"""
Build a Variational Autoencoder, in a variational Autoencoder we wish to
generate new data points from the vector space of our original data,
we want the generated data to be similar but not the same.
"""

import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Build a Variational Autoencoder.

    Args:
        input_dims : input dimension which is the same as output
        hidden_layers: list number of nodes in each hidden layer
        latent_dims: dimension of the latent space layer

    Returns:
        Tuple of encoder, decoder, and autoencoder keras models
    """
    # The encoder
    inputs = K.Input(shape=(input_dims,))
    encoded = inputs
    for layer_size in hidden_layers:
        encoded = K.layers.Dense(layer_size, activation='relu')(encoded)

    z_mean = K.layers.Dense(latent_dims, name='z_mean')(encoded)
    z_log_var = K.layers.Dense(latent_dims, name='z_log_var')(encoded)

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.backend.shape(z_mean)[0]
        dim = K.backend.int_shape(z_mean)[1]
        epsilon = K.backend.random_normal(shape=(batch, dim))
        return z_mean + K.backend.exp(0.5 * z_log_var) * epsilon

    z = K.layers.Lambda(sampling, name='z')([z_mean, z_log_var])
    encoder = K.Model(inputs, [z, z_mean, z_log_var], name='encoder')

    # The Decoder
    latent_inputs = K.Input(shape=(latent_dims,))
    decoded = latent_inputs
    for layer_size in reversed(hidden_layers):
        decoded = K.layers.Dense(layer_size, activation='relu')(decoded)

    outputs = K.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = K.Model(latent_inputs, outputs, name='decoder')

    # The AutoEncoder
    outputs = decoder(encoder(inputs)[0])
    autoencoder = K.Model(inputs, outputs, name='autoencoder')

    def vae_loss(inputs, outputs):
        reconstruction_loss = K.losses.binary_crossentropy(inputs, outputs) * input_dims
        kl_loss = - 0.5 * K.backend.mean(1 + z_log_var - K.backend.square(z_mean) - K.backend.exp(z_log_var), axis=-1)
        return K.backend.mean(reconstruction_loss + kl_loss)

    autoencoder.compile(optimizer='adam', loss=vae_loss)
    return encoder, decoder, autoencoder
