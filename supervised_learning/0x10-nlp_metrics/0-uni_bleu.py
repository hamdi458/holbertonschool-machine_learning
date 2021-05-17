#!/usr/bin/env python3
""" unigram blue score """
import numpy as np


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence"""
    ref = []
    words = {}

    for i in references:
        ref.append(len(i))
        for w in i:
            if w in sentence:
                if not words.keys() == w:
                    words[w] = 1
    prob = sum(words.values())
    indice = np.argmin([abs(len(x) - len(sentence)) for x in references])
    best_match = len(references[indice])
    if len(sentence) > best_match:
        bp = 1
    else:
        bp = np.exp(1 - float(best_match) / float(len(sentence)))
    Blue_score = bp * np.exp(np.log(prob / len(sentence)))
    return Blue_score
