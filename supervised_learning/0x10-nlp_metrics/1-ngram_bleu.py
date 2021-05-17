#!/usr/bin/env python3
""" n-gram blue score """
import numpy as np


def build_n_gram(sentence, ind):
    """Function that creates an ngram"""
    s = []
    for i in range(len(sentence) - ind + 1):
        n_gram = ""
        for j in range(ind):
            n_gram += sentence[i + j]
            if not j + 1 == ind:
                n_gram += " "
        s.append(n_gram)
    return s


def ngram_bleu(references, sentence, n):
    """Function that calculates the n-gram BLEU score for a sentence"""
    r_list = np.array([abs(len(ref) - len(sentence)) for ref in references])
    mask = np.where(r_list == r_list.min())
    r = np.array([len(ref) for ref in references])[mask].sum()
    cn = build_n_gram(sentence, n)
    candidate = {x: 0 for x in cn}
    references = [build_n_gram(ref, n) for ref in references]
    max_match = 0
    for ref in references:
        match = 0
        ref_dict = {x: ref.count(x) for x in ref}
        for key in ref_dict.keys():
            if key in candidate:
                match += 1
            if match > max_match:
                max_match = match
    P = max_match / len(cn)
    if len(sentence) <= r:
        BP = np.exp(1-(r / len(sentence)))

    else:
        BP = 1

    Bp = BP * P
    return Bp
