#!/usr/bin/env python3
"""gensim FastText model"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """creates and trains a genism fastText model:"""
    fasttext_model = FastText(sentences=sentences, min_count=min_count,
                              iter=iterations,
                              size=size, window=window, sg=cbow, seed=seed,
                              negative=negative, workers=workers)
    fasttext_model.train(sentences=sentences,
                         total_examples=model.corpus_count,
                         epochs=iterations)
    return fasttext_model
