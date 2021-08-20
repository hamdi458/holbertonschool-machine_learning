#!/usr/bin/env python3
"""1. TD(λ)"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """function that performs the TD(λ) algorithm"""
    episode = [[], []]
    x = [0 for i in range(env.observation_space.n)]
    for jdx in range(episodes):
        st = env.reset()
        for idx in range(max_steps):
            x = list(np.array(x) * lambtha * gamma)
            x[st] += 1
            action = policy(st)
            new_state, reward, d, info = env.step(action)
            delta_t = reward + gamma * V[new_state] - V[st]
            V[st] = V[st] + alpha * delta_t * x[st]
            if d != 0:
                break
            st = new_state
    return np.array(V)
