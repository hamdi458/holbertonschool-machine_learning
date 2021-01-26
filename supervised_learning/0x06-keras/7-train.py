#!/usr/bin/env python3
"""train"""
import tensorflow.keras as k


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """train"""
    if learning_rate_decay and validation_data:
        def decayed_learning_rate(epochs):
            return alpha / (1 + decay_rate * epochs)
        lrate = k.callbacks.LearningRateScheduler(decayed_learning_rate, 1)
        callbacks_list = [lrate]
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, callbacks=callbacks_list,
                              verbose=verbose, validation_data=validation_data)
    if early_stopping and validation_data:
        stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                         mode='min',
                                         patience=patience)
        callbacks_stop_list = [stop]

        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, callbacks=callbacks_stop_list,
                              verbose=verbose, validation_data=validation_data)
    if validation_data is None:
        callback = None

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        shuffle=shuffle,
        verbose=verbose,
        callbacks=callback)
    return history
