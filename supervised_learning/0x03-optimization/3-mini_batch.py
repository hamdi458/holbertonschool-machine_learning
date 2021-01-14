#!/usr/bin/env python3
"""Write the function that trains a loaded neural network model
using mini-batch gradient descent:"""
import tensorflow as tf


shuffle_data = __import__('2-shuffle_data').shuffle_data


def cat(datax, datay, batch_size):
    """split data"""
    arrdatax = []
    arrdatay = []
    m = datax.shape[0]

    num = m / batch_size
    if num % 1 != 0:
        num = int(num + 1)
    else:
        num = int(num)

    for i in(range(num)):
        if i != num - 1:
            arrdatax.append(datax[i*batch_size:((i+1)*batch_size)])
            arrdatay.append(datay[i*batch_size:((i+1)*batch_size)])
        else:
            arrdatax.append(datax[i*batch_size:])
            arrdatay.append(datay[i*batch_size:])
    return arrdatay, arrdatax


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """trains a loaded neural network model
    using mini-batch gradient descent"""

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.import_meta_graph('{}.meta'.format(load_path))
        saver.restore(sess, '{}'.format(load_path))

        x = tf.get_collection('x', scope=None)[0]
        y = tf.get_collection('y', scope=None)[0]
        loss = tf.get_collection('loss', scope=None)[0]
        accuracy = tf.get_collection('accuracy', scope=None)[0]
        train_op = tf.get_collection('train_op', scope=None)[0]
        for epoche in range(epochs+1):
            X_shuffled_train, Y_shuffled_train = shuffle_data(X_train, Y_train)

            train_loss = sess.run(loss, {
                x: X_train, y: Y_train})
            train_acc = sess.run(accuracy, {
                x: X_train, y: Y_train})
            valid_acc = sess.run(accuracy, {
                x: X_valid, y: Y_valid})

            valid_loss = sess.run(loss, {x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoche))
            print("\tTraining Cost: {}".format(train_loss))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_loss))
            print("\tValidation Accuracy: {}".format(
                valid_acc))
            if epoche != epochs:
                arrx, arry = cat(
                             Y_shuffled_train, X_shuffled_train, batch_size)
                for i in range(len(arrx)):
                    sess.run(train_op, {x: arrx[i], y: arry[i]})
                    train_loss = sess.run(loss, {x: arrx[i], y: arry[i]})
                    train_acc = sess.run(accuracy, {x: arrx[i],
                                         y: arry[i]})
                    if((i + 1) % 100 == 0 and i > 0):
                        print("\tStep {}:".format(i+1))
                        print("\tTraining Cost: {}".format(train_loss))
                        print("\tTraining Accuracy: {}".format(train_acc))
        return saver.save(sess, save_path)
