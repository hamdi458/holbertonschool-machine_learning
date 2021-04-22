#!/usr/bin/env python3
"""Program that creates a variational autoencoder"""
import tensorflow.keras as keras


def sample(args):
    """samling a new points"""
    z_mean, z_log_sigma = args
    batch = keras.backend.shape(z_mean)[0]
    dims = keras.backend.int_shape(z_mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dims))
    return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Function that creates a variational autoencoder"""
    X = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(hidden_layers[0], activation='relu')(X)
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoded)
    z_mean = keras.layers.Dense(latent_dims)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims)(encoded)
    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoder = keras.Model(X, [z, z_mean, z_log_sigma])
    latentX = keras.Input(shape=(latent_dims,))
    decoded = latentX
    for i in range(len(hidden_layers) - 1, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(latentX, decoded)

    def loss(true, pred):
        """Rexonatruct loss"""
        reconstruction_loss = keras.losses.binary_crossentropy(X,
                                                               vae_outputs)
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) -\
            keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return keras.backend.mean(reconstruction_loss + kl_loss)
    vae_outputs = decoder(encoder(X))
    auto = keras.Model(X, vae_outputs)
    auto.compile(optimizer='adam', loss=loss)
    return encoder, decoder, auto
