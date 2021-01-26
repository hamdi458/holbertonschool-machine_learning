#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *


def optimize_model(network, alpha, beta1, beta2):
    opt = keras.optimizers.Adam(alpha, beta1, beta2)
    network.compile(loss='categorical_crossentropy', optimizer=opt,
                    metrics=['accuracy'])
    return None
