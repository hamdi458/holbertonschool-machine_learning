#!/usr/bin/env python3
"""0. Monte Carlo"""
import numpy as np
import gym


def generate_episode(env, policy, max_steps):
    """generate list of tuple  state and reward"""
    state = env.reset()
    episode = []
    for i in range(max_steps):
        action = policy(state)
        next_state, reward, d, _ = env.step(action)
        episode.append((state, reward))
        state = next_state
        if d:
            break
    return episode


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
   function def monte_carlo(env, V, policy, episodes=5000,
            max_steps=100, alpha=0.1, gamma=0.99):
   that performs the Monte Carlo algorithm:
    """
    returns = set()
    for i in range(1, episodes+1):
        episode = generate_episode(env, policy, max_steps)
        # Update Q values
        states, rewards = zip(*episode)
        discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])

        for index in range(len(episode[0])-1, -1, -1):

            G = sum(rewards[index:] * discounts[:-(index+1)])
            if not episode[index][0] in returns:
                V[episode[index][0]] = V[episode[index][0]] + \
                    alpha * (G - V[episode[index][0]])
            returns.add(episode[index][0])
    return V
