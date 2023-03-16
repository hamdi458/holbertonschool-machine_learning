#!/usr/bin/env python3
import tensorflow.keras as K

def autoencoder(input_dims, hidden_layers, latent_dims):
    # Encoder
    enc_in = K.layers.Input(shape=(input_dims,))
    x = enc_in
    for layer_size in hidden_layers:
        x = K.layers.Dense(layer_size, activation="relu")(x)

    z_mean = K.layers.Dense(latent_dims)(x)
    z_log_var = K.layers.Dense(latent_dims)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.backend.random_normal(shape=K.backend.shape(z_mean))
        return z_mean + K.backend.exp(z_log_var / 2) * epsilon

    z = K.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = K.Model(enc_in, [z, z_mean, z_log_var])

    # Decoder
    dec_in = K.layers.Input(shape=(latent_dims,))
    x = dec_in
    for layer_size in reversed(hidden_layers):
        x = K.layers.Dense(layer_size, activation="relu")(x)
    x = K.layers.Dense(input_dims, activation="sigmoid")(x)
    decoder = K.Model(dec_in, x)

    # Instantiate VAE model
    outputs = decoder(encoder(enc_in)[0])
    vae = K.Model(enc_in, outputs)

    # Define loss function
    def vae_loss(x, x_decoded_mean):
        xent_loss = K.backend.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.backend.mean(1 + z_log_var - K.backend.square(z_mean) - K.backend.exp(z_log_var), axis=-1)
        return K.backend.mean(xent_loss + kl_loss)

    # Compile VAE model
    vae.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, vae
