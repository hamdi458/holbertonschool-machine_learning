#!/usr/bin/env python3
"""gensim word2vec model"""


def gensim_to_keras(model):
    """onverts a gensim word2vec model to a keras Embedding layer"""
    return model.wv.get_keras_embedding(train_embeddings=True)