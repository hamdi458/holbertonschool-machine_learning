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
                                    split='validation', as_supervised=True)
        a, b = self.tokenize_dataset(self.data_train)
        self.tokenizer_en = b
        self.tokenizer_pt = a

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for our dataset"""
        pp = []
        ee = []
        for pt, en in data:
            pp.append(pt.numpy())
            ee.append(en.numpy())
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pp, target_vocab_size=2**15)
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            ee, target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """encode function"""
        tok_pt = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(pt.numpy()) + [(self.tokenizer_pt.vocab_size) + 1]
        tok_en = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(en.numpy()) + [(self.tokenizer_en.vocab_size) + 1]
        return tok_pt, tok_en
