#!/usr/bin/env python3
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Build a Variational Autoencoder.
    Args:
        input_dims : input dimension which is the same as output
        hidden_layers: list number of nodes in each hidden layer
        latent_dims: dimension of the latent space layer
    Return:
        Model: the autoencoder keras model
    """
    # The encoder
    X = keras.Input((input_dims,))
    en = X
    for nl in hidden_layers:
        en = keras.layers.Dense(nl, activation='relu')(en)
    elogvar = keras.layers.Dense(latent_dims, activation=None)(en)
    emean = keras.layers.Dense(latent_dims, activation=None)(en)

    def sampling(params):
        mu, logvar = params
        epsilon = keras.backend.random_normal(shape=(keras.backend.shape(mu)[0], latent_dims), mean=0.0, stddev=1.0)
        return mu + keras.backend.exp(0.5 * logvar) * epsilon

    z = keras.layers.Lambda(sampling)((emean, elogvar))
    encoder = keras.Model(X, [z, emean, elogvar])

    # The Decoder
    de_X = keras.Input((latent_dims,))
    de = de_X
    for nl in hidden_layers[::-1]:
        de = keras.layers.Dense(nl, activation='relu')(de)
    de_final = keras.layers.Dense(input_dims, activation='sigmoid')(de)
    decoder = keras.Model(de_X, de_final)

    # The AutoEncoder
    enc, mu, logvar = encoder(X)

    dec = decoder(enc)
    auto = keras.Model(X, dec)

    def vae_loss(y_true, y_pred):
        reconstruction_loss = keras.losses.binary_crossentropy(y_true, y_pred) * input_dims
        kl_loss = -0.5 * keras.backend.mean(1 + elogvar - keras.backend.square(emean) - keras.backend.exp(elogvar), axis=-1)
        vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
        return vae_loss

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
