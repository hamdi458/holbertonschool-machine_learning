#!/usr/bin/env python3
"""ResNet-50"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """builds the ResNet-50 architecture"""
    input = K.layers.Input(shape=(224, 224, 3))
    kernel = K.initializers.he_normal(seed=None)
    output = K.layers.Conv2D(kernel_size=(7, 7),
                             strides=2, padding='same',
                             kernel_initializer=kernel,
                             filters=64)(input)
    output = K.layers.BatchNormalization(axis=3)(output)
    output = K.layers.Activation('relu')(output)
    output = K.layers.MaxPooling2D((3, 3), strides=(2, 2))(output)

    output = projection_block(output, [64, 64, 256], s=1)
    output = identity_block(output, [64, 64, 256])
    output = identity_block(output, [64, 64, 256])

    output = projection_block(output, [128, 128, 512])
    output = identity_block(output, [128, 128, 512])
    output = identity_block(output, [128, 128, 512])
    output = identity_block(output, [128, 128, 512])

    output = projection_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])

    output = projection_block(output, [512, 512, 2048])
    output = identity_block(output, [512, 512, 2048])
    output = identity_block(output, [512, 512, 2048])

    output = K.layers.GlobalAveragePooling2D()(output)
    classes = 1000
    output = K.layers.Dense(classes, activation='softmax')(output)
    model = K.models.Model(inputs=input, outputs=output)
    return model
