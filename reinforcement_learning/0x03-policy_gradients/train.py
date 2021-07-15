#!/usr/bin/env python3
"""
Training reinforcement model
"""
import numpy as np
import matplotlib.pyplot as plt


def policy_gradient(state, prb, action):
    """Function that computes the Monte-Carlo policy gradient"""
    def softmax_grad(softmax):
        s = softmax.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    dsoftmax = softmax_grad(prb)[action, :]
    dlog = dsoftmax / prb[0, action]
    gradient = state.T.dot(dlog[None, :])

    return gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Function that implements a full training"""
    w = np.random.rand(4, 2)
    nA = env.action_space.n
    num_ep = []
    for e in range(nb_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        score = 0

        while 1:
            if show_result and e % 1000 == 0:
                env.render()
            z = state.dot(w)
            exp = np.exp(z)

            probs = exp / np.sum(exp)
            action = np.random.choice(nA, p=probs[0])
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[None, :]
            grad = policy_gradient(state, probs, action)
            grads.append(grad)
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state
            if done is True:
                break
        for i in range(len(grads)):
            aux = sum([r * (gamma ** r) for t, r in enumerate(rewards[i:])])
            w += alpha * grads[i] * aux
        num_ep.append(score)
        print("EP: " + str(e) + " Score: " + str(score) + "        ",
              end="\r", flush=False)
    return num_ep
