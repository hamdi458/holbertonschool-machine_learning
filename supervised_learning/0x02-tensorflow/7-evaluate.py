#!/usr/bin/env python3
"""evaluates the output of a neural network:"""

import tensorflow as tf
import numpy as np


def evaluate(X, Y, save_path):
    """evaluates the output of a neural network:"""
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.import_meta_graph('{}.meta'.format(save_path))
        x = tf.get_collection('x', scope=None)[0]
        y = tf.get_collection('y', scope=None)[0]
        y_pred = tf.get_collection('y_pred', scope=None)[0]
        loss = tf.get_collection('loss', scope=None)[0]
        accuracy = tf.get_collection('accuracy', scope=None)[0]
        train_op = tf.get_collection('train_op', scope=None)[0]
        accuracy = sess.run(accuracy, {x: X, y: Y})
        j = sess.run(loss, {x: X, y: Y})
        y_pred = sess.run(y_pred, {x: X, y: Y})

        return y_pred, accuracy, j
