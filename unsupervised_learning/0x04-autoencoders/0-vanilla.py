#!/usr/bin/env python3
"""Vanilla Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates an autoencoder"""
    X = keras.Input(shape=(input_dims,))
    encoder = keras.layers.Dense(hidden_layers[0], activation='relu')(X)
    for i in range(1, len(hidden_layers)):
        encoder = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoder)
    encoder = keras.layers.Dense(latent_dims, activation='relu')(encoder)
    encoder = keras.Model(X, encoder)
    decoder_input = keras.Input(shape=(latent_dims,))
    decoder = decoder_input
    for i in range(len(hidden_layers) - 1, -1, -1):
        decoder = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(decoder)
    decoder = keras.layers.Dense(input_dims, activation='sigmoid')(decoder)
    decoder = keras.Model(decoder_input, decoder)
    auto = keras.Model(X, decoder(encoder(X)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
