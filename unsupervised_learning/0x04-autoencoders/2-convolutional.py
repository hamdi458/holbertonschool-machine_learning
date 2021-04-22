#!/usr/bin/env python3
"""Sparse autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """creates a sparse autoencoder"""
    X = keras.Input(shape=(input_dims))
    conv = X
    for nb_f in filters:
        conv = keras.layers.Conv2D(nb_f, (3, 3), activation='relu',
                                      padding='same')(conv)
        conv = keras.layers.MaxPooling2D((2, 2), padding='same')(conv)
    encoder = keras.Model(X, conv)
    zoujeja = keras.Input(shape=(latent_dims))
    dec = zoujeja
    i = 0
    for nb_ff in reversed(filters):
        if (i == len(filters) - 1):
            dec = keras.layers.Conv2D(nb_ff, (3, 3), activation='relu',
                                    padding='valid')(dec)
        else:
            dec = keras.layers.Conv2D(nb_ff, (3, 3), activation='relu',
                                    padding='same')(dec)
        dec = keras.layers.UpSampling2D((2, 2))(dec)
        
        if (i == len(filters)):
            break
        i = i+1

    decoder = keras.layers.Conv2D(input_dims[2], (3, 3), activation='sigmoid',
                                  padding='same')(dec)

    decoder = keras.Model(zoujeja, decoder)
    auto = keras.Model(X, decoder(encoder(X)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto