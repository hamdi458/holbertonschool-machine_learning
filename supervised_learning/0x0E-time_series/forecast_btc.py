#!/usr/bin/env python3
import tensorflow.keras as k
import numpy as np
import tensorflow.keras as K
import pandas as pd
import datetime as dt
import tensorflow as tf

class WindowGenerator:
    """WindowGenerator Class"""

    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        """initialize"""

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = \
            np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = \
            np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        """split_window function"""
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                 for name in self.label_columns],
                axis=-1)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='Close', max_subplots=3):
        """Plotting"""

        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_idx = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel('{} [normed]'.format(plot_col))
            plt.plot(self.input_indices, inputs[n, :, plot_col_idx],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_idx = self.label_columns_indices.get(plot_col,
                                                               None)
            else:
                label_col_idx = plot_col_idx

            if label_col_idx is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_idx],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                prediction = model(inputs)
                plt.scatter(self.label_indices,
                            prediction[n, :, label_col_idx],
                            marker='X', edgecolors='k',
                            label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        """
        Convert to tf.dataset
        Argumetns:
         - data
        Returns:
         ds as tf.dataset
        """

        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds
class Baseline(tf.keras.Model):
    """
    Baseline class
    """

    def __init__(self, label_index=None):
        """
        Class constructor
        Argumetns:
         - label_index
        """

        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        """
        call
        Argumetns:
         - param inputs
        """
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]



def compile_and_fit(model, window, patience=2, epochs=500):
    """compile and fit model"""
    e_s = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=patience,
                                           mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    model = model.fit(window.train_df,window.val_df ,epochs=epochs,
                        callbacks=[e_s])
    print(model.summary())


    return model

def forecasting(train, validation, test):
    """
    Function for forecasting model of the BTC price
    Arguments:
     - train is the train values
     - validation is the validation values
     - test is the test values
    """

    window = WindowGenerator(input_width=24, label_width=24, shift=1,
                             train_df=train, val_df=validation, test_df=test,
                             label_columns=['Close'])
    column_indices = window.column_indices
    print(window)

    val_performance = {}
    performance = {}

    baseline = Baseline(label_index=column_indices['Close'])

    baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                     metrics=[tf.keras.metrics.MeanAbsoluteError()])

    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(units=1)])
    history = compile_and_fit(lstm_model, window)

    val_performance['LSTM'] = lstm_model.evaluate(window.val)
    performance['LSTM'] = lstm_model.evaluate(window.test, verbose=0)
    window.plot(lstm_model)