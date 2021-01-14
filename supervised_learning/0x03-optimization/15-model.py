#!/usr/bin/env python3
"""Adam optimization, mini-batch gradient descent,
learning rate decay, and batch normalization"""
import tensorflow as tf
import numpy as np


def create_placeholders(nx, classes):
    """returns two placeholders, x and y, for the neural network"""
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")
    return x, y


def create_layer(prev, n, activation):
    """ create layer"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=kernel, name="layer",
                            )
    return layer(prev)


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""
    i = np.random.permutation(np.arange(X.shape[0]))
    return X[i], Y[i]


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """creates a learning rate decay operation in tensorflow
    using inverse time decay"""
    t = tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True)
    return t


def create_batch_norm_layer(prev, n, activation):
    """batch normalization layer for a neural network in tensorflow"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=None,
                            kernel_initializer=kernel, name="layer",
                            )

    z = layer(prev)

    batch_mean1, batch_var1 = tf.nn.moments(z, [0])
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    a = tf.nn.batch_normalization(z, mean=batch_mean1, variance=batch_var1,
                                  offset=beta, scale=gamma,
                                  variance_epsilon=1e-8)
    return activation(a)


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction:"""
    pred = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(pred, tf.float32))


def calculate_loss(y, y_pred):
    """calculates the softmax cross-entropy loss of a prediction"""
    return tf.losses.softmax_cross_entropy(y, y_pred)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """training operation for a neural network in tensorflow
    using the Adam optimization algorithm"""
    minimize = tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        use_locking=False,
        name='Adam')
    return minimize.minimize(loss)


def forward_prop(x, layer_sizes=[], activations=[]):
    """forward propagation graph for the neural network"""
    for i, act in zip(layer_sizes, activations):
        if act is None:
            y = create_layer(x, i, act)
            x = y
        else:
            y = create_batch_norm_layer(x, i, act)
            x = y
    return y


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


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """ builds, trains, and saves a neural network model in tensorflow
    using Adam optimization, mini-batch gradient descent,
    learning rate decay, and batch normalization:"""
    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layers, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    global_step = tf.Variable(0, trainable=False)
    decay = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, decay, beta1, beta2, epsilon)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for ep in range(epochs + 1):
            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)
            loss_train = sess.run(loss, {x: X_train, y: Y_train})
            acc_train = sess.run(accuracy, {x: X_train, y: Y_train})
            loss_valid = sess.run(loss, {x: X_valid, y: Y_valid})
            acc_valid = sess.run(accuracy, {x: X_valid, y: Y_valid})
            print('After {} epochs:'.format(ep))
            print('\tTraining Cost: {}'.format(loss_train))
            print('\tTraining Accuracy: {}'.format(acc_train))
            print('\tValidation Cost: {}'.format(loss_valid))
            print('\tValidation Accuracy: {}'.format(acc_valid))

            if ep != epochs:
                Y_batch, X_batch = cat(X_shuffle, Y_shuffle, batch_size)
                for i in range(1, len(X_batch) + 1):
                    sess.run(train_op, {x: X_batch[i - 1], y: Y_batch[i - 1]})

                    loss_train = sess.run(loss, {x: X_batch[i-1],
                                                 y: Y_batch[i-1]})
                    acc_train = sess.run(accuracy, {x: X_batch[i-1],
                                                    y: Y_batch[i-1]})

                    if(i % 100 == 0):
                        print('\tStep {}:'.format(i))
                        print('\t\tCost: {}'.format(loss_train))
                        print('\t\tAccuracy: {}'.format(acc_train))

        save_path = saver.save(sess, save_path)
    return save_path
