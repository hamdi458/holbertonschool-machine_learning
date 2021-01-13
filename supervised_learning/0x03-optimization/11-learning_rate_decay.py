#!/usr/bin/env python3
"""function def learning_rate_decay(alpha, decay_rate, global_step,
decay_step):
that updates the learning rate using inverse time decay in numpy:"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """updates the learning rate using inverse time decay in numpy"""
    alpha = alpha / (1 + (decay_rate * global_step // decay_step))
    return alpha
