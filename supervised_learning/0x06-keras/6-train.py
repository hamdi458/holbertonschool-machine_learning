#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *

def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False):
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=patience)
    if early_stopping and validation_data:
        history = network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle, validation_data=validation_data, callbacks=[es])
    else:
        history = network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle, validation_data=validation_data)

    return history