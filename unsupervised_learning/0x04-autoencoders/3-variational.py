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

    def sampler(params):
        mu, logstd = params
        rand = keras.backend.random_normal((keras.backend.shape(mu)[0],
                                           latent_dims), mean=0,
                                           stddev=1)
        return mu + keras.backend.exp(logstd / 2) * rand
    en_final = keras.layers.Lambda(sampler)((emean, elogvar))
    encoder = keras.Model(X, [en_final, emean, elogvar])

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

    def vae_loss(true, pred):
        """calculate loss"""
        reconstruction_loss = K.losses.binary_crossentropy(enc_in, outputs)
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma - K.backend.square(z_mean) - \
            K.backend.exp(z_log_sigma)
        kl_loss = K.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.backend.mean(reconstruction_loss + kl_loss)


    # auto.add_loss(bce_loss)
    # auto.compile(optimizer='adam')
    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
