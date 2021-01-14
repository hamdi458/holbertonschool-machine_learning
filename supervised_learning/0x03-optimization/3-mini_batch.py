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

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]

        for i in range(epochs + 1):
            X, Y = shuffle_data(X_train, Y_train)
            loss_tr, acc_tr = sess.run((loss, accuracy),
                                       feed_dict={x: X_train, y: Y_train})
            loss_v, acc_v = sess.run((loss, accuracy),
                                     feed_dict={x: X_valid, y: Y_valid})
            print('After {} epochs:'.format(i))
            print('\tTraining Cost: {}'.format(loss_tr))
            print('\tTraining Accuracy: {}'.format(acc_tr))
            print('\tValidation Cost: {}'.format(loss_v))
            print('\tValidation Accuracy: {}'.format(acc_v))

            if epoche != epochs:	
                Y_bathc, X_batch = cat(X,Y, batch_size)
                for b in range(1, len(X_batch) + 1):
                    sess.run(train_op, feed_dict={x: X_batch[b - 1],
                                                  y: Y_bathc[b - 1]})

                    loss_tr, acc_tr = sess.run((loss, accuracy),
                                               feed_dict={x: X_batch[b - 1],
                                                          y: Y_bathc[b - 1]})

                    if(b % 100 == 0):	
                        print('\tStep {}:'.format(b))
                        print('\t\tCost: {}'.format(loss_tr))
                        print('\t\tAccuracy: {}'.format(acc_tr))

        save_path = saver.save(sess, save_path)

    return save_path