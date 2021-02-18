#!/usr/bin/env python3
"""transfer learning resnet50"""
import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """trains a convolutional neural network to classify the dataset"""
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


(trainX, trainy), (testX, testy) = K.datasets.cifar10.load_data()
trainX, trainy = preprocess_data(trainX, trainy)
testX, testy = preprocess_data(testX, testy)
inputs = K.Input(shape=(224, 224, 3))

"""Loading the ResNet50 model with pre-trained ImageNet weights
"""
resnet = K.applications.ResNet50(weights='imagenet',
                                 include_top=False, input_tensor=inputs)

for layer in resnet.layers[:170]:
    layer.trainable = False

model = K.models.Sequential()
model.add(K.layers.Lambda(lambda x: tf.image.resize(x, (224, 224))))
model.add(resnet)
model.add(K.layers.GlobalAveragePooling2D())
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(256, activation='relu'))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(128, activation='relu'))
model.add(K.layers.Dropout(0.3))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(64, activation='relu'))
model.add(K.layers.Dropout(0.3))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(10, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer=K.optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
checkpointer = K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                                           monitor="val_accuracy",
                                           verbose=1, save_best_only=True)

model.fit(trainX, trainy, batch_size=32, epochs=10,
          verbose=1, callbacks=[checkpointer],
          validation_data=(testX, testy), shuffle=True)
model.summary()
model.save("cifar10.h5")
