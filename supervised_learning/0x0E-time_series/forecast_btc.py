#!/usr/bin/env python3
import tensorflow.keras as k


def model(x, y):
    """build rnn model (ltms)"""
    regressor = k.models.Sequential()
    regressor.add(k.layers.LSTM(units = 24, activation = 'sigmoid', input_shape = [24, 7], return_sequences=True))
    # Adding the output layer

    regressor.add(Dense(units = 1))
    # compile the model
    return regressor

def 
