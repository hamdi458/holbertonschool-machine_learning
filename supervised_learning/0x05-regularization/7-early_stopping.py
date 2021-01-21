#!/usr/bin/env python3
"""function def early_stopping(cost, opt_cost, threshold, patience, count):
that determines if you should stop gradient descent early:"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """determines if you should stop gradient descent early"""
    if opt_cost - cost <= threshold:
        count += 1
    else:
        count = 0
    return patience == count, count
