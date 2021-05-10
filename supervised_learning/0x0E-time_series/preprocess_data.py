#!/usr/bin/env python3
"""Preprocessing the database"""
import pandas as pd


def preprocessing(csv_path):
    df = pd.read_csv(csv_path)
    # drop nan
    df = df.dropna()
    # take the last 2 years data
    df = df[-(730 * 24 * 60):]
    # encode the date
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
    # make date as index
    df = df.set_index('Date')
    # you should always split your data into training, validation, testing
    # (usually 80%, 10%, 10%)
    df_train = df[:int(len(df)*80/100)]
    df_valid = df[int(len(df_train)):int(len(df)*90/100)]
    df_test = df[int(-(len(df)*10/100)):]

    # normalize data
    train_mean = df_train.mean()
    train_std = df_train.std()
    x_train = (df_train - train_mean) / train_std
    x_valid = (df_valid - train_mean) / train_std
    x_test = (df_test - train_mean) / train_std
    return x_train, x_valid, x_test
