#!/usr/bin/env python3
"""function
def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
that updates a variable in place using the Adam optimization algorithm:"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """updates a variable in place using the Adam optimization algorithm"""
    vdw = beta1 * v + (1 - beta1)*grad
    sdw = beta2 * s + (1-beta2)*grad**2
    vdw_c = vdw/(1-beta1**t)
    sdw_c = sdw/(1-beta2**t)
    var = var - alpha*vdw_c/(sdw_c**0.5+epsilon)
    return var, vdw, sdw
