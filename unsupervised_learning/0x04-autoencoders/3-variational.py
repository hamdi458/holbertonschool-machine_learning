#!/usr/bin/env python3
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    """
    Build a Variational Autoencoder.
    Build a Variational Autoencoder.
    Args:
    Args:
        input_dims : input dimension which is the same as output
        input_dims : input dimension which is the same as output
        hidden_layers: list number of nodes in each hidden layer
        hidden_layers: list number of nodes in each hidden layer
        latent_dims: dimension of the latent space layer
        latent_dims: dimension of the latent space layer
    Return:
    Return:
        Model: the autoencoder keras model
        Model: the autoencoder keras model
    """
    """
    # The encoder
    # The encoder
    X = keras.Input((input_dims,))
    X = keras.Input((input_dims,))
    en = X
    en = X
    for nl in hidden_layers:
    for nl in hidden_layers:
        en = keras.layers.Dense(nl, activation='relu')(en)
        en = keras.layers.Dense(nl, activation='relu')(en)
    elogvar = keras.layers.Dense(latent_dims, activation=None)(en)
    elogvar = keras.layers.Dense(latent_dims, activation=None)(en)
    emean = keras.layers.Dense(latent_dims, activation=None)(en)
    emean = keras.layers.Dense(latent_dims, activation=None)(en)
    def sampler(params):
    def sampler(params):
        mu, logstd = params
        mu, logstd = params
        rand = keras.backend.random_normal((keras.backend.shape(mu)[0],
        rand = keras.backend.random_normal((keras.backend.shape(mu)[0],
                                           latent_dims), mean=0,
                                           latent_dims), mean=0,
                                           stddev=1)
                                           stddev=1)
        return mu + keras.backend.exp(logstd / 2) * rand
        return mu + keras.backend.exp(logstd / 2) * rand
    en_final = keras.layers.Lambda(sampler)((emean, elogvar))
    en_final = keras.layers.Lambda(sampler)((emean, elogvar))
    encoder = keras.Model(X, [en_final, emean, elogvar])
    encoder = keras.Model(X, [en_final, emean, elogvar])
    # The Decoder
    # The Decoder
    de_X = keras.Input((latent_dims,))
    de_X = keras.Input((latent_dims,))
    de = de_X
    de = de_X
    for nl in hidden_layers[::-1]:
    for nl in hidden_layers[::-1]:
        de = keras.layers.Dense(nl, activation='relu')(de)
        de = keras.layers.Dense(nl, activation='relu')(de)
    de_final = keras.layers.Dense(input_dims, activation='sigmoid')(de)
    de_final = keras.layers.Dense(input_dims, activation='linear')(de)
    decoder = keras.Model(de_X, de_final)
    decoder = keras.Model(de_X, de_final)
    # The AutoEncoder
    # The AutoEncoder
    enc, mu, logvar = encoder(X)
    enc, mu, logvar = encoder(X)
    dec = decoder(enc)
    dec = decoder(enc)
    auto = keras.Model(X, dec)
    auto = keras.Model(X, dec)
    def vae_loss(y_true, y_pred):
    def vae_loss(y_true, y_pred):
        bce_loss = keras.losses.binary_crossentropy(y_pred,
        mse_loss = keras.losses.mean_squared_error(y_pred, y_true) * input_dims
                                                    y_true) * input_dims
        kl_loss = keras.backend.sum(1 + logvar - keras.backend.square(mu) -
        kl_loss = keras.backend.sum(1 + logvar - keras.backend.square(mu) -
                                    keras.backend.exp(logvar), axis=-1) * -0.5
                                    keras.backend.exp(logvar), axis=-1) * -0.5
        vae_loss = keras.backend.mean(bce_loss + kl_loss)
        vae_loss = keras.backend.mean(mse_loss + kl_loss)
        return vae_loss
        return vae_loss
    # auto.add_loss(bce_loss)
    # auto.compile(optimizer='adam')
    auto.compile(optimizer='adam', loss=vae_loss)
    auto.compile(optimizer='adam', loss=vae_loss)
    return encoder, decoder, auto
    return encoder, decoder, auto
