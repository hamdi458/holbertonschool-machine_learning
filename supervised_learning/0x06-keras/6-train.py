#!/usr/bin/env python3
""" train the model using early stopping"""
import tensorflow.keras as k


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ train the model using early stopping:"""
    es = k.callbacks.EarlyStopping(monitor='val_loss',
                                   mode='min', verbose=verbose,
                                   patience=patience)
    if early_stopping and validation_data:
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle, validation_data=validation_data,
                              callbacks=[es])
    else:
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose, shuffle=shuffle,
                              validation_data=validation_data)

    return history
