#!/usr/bin/env python3
"""train"""
import tensorflow.keras as k


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                save_best=False, filepath=None, decay_rate=1, verbose=True,
                shuffle=False):
    """train"""
    if learning_rate_decay and validation_data:
        def decayed_learning_rate(epochs):
            return alpha / (1 + decay_rate * epochs)
        lrate = k.callbacks.LearningRateScheduler(decayed_learning_rate,
                                                  verbose=1)
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
    if save_best and validation_data:
        checkpoint = k.callbacks.ModelCheckpoint(filepath=filepath,
                                                 monitor='val_loss',
                                                 save_best_only=True)
        callbacks_list_save = [checkpoint]
        history = network.fit(x=data, y=labels, epochs=epochs,
                              verbose=verbose,
                              batch_size=batch_size,
                              validation_data=validation_data,
                              shuffle=shuffle, callbacks=callbacks_list_save)
    if validation_data is None:
        history = network.fit(
            data,
            labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            shuffle=shuffle,
            verbose=verbose)
    return history
