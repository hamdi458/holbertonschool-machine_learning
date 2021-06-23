#!/usr/bin/env python3
"""Initialize Q-table"""
import numpy as np


def q_init(env):
    """initializes the Q-table"""
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    return np.zeros((state_space_size, action_space_size))
