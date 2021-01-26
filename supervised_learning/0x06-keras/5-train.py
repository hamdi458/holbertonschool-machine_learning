#!/usr/bin/env python3

def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False):
    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle, validation_data=validation_data)
    return history