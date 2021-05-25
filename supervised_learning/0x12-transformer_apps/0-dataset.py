#!/usr/bin/env python3
"""Dataset"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Dataset class"""
    def __init__(self):
        """initialize class constructor"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)git

        a, b = self.tokenize_dataset(self.data_train)
        self.tokenizer_en = a
        self.tokenizer_pt = b

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for our dataset"""
        pp = []
        ee = []
        for pt, en in data:
            pp += (pt.numpy()).split()
            ee += (en.numpy()).split()
        encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        pp, target_vocab_size=2**15)
        encoder2 = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        ee, target_vocab_size=2**15)
        return encoder , encoder2
