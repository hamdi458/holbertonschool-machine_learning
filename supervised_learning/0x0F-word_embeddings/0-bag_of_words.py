#!/usr/bin/env python3
"""bag of wards"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix"""
    if vocab is None:
        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names()
    else:
        vectorizer = CountVectorizer(vocabulary=vocab)
        Xx = vectorizer.fit_transform(sentences)
    return x.toarray(), vocab
