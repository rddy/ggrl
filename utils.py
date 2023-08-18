import random
import os

import numpy as np
import torch
import gymnasium as gym


DATA_DIR = "data"
MODEL_DIR = os.path.join(DATA_DIR, "models")
EXP_DIR = os.path.join(DATA_DIR, "exp")
for path in [DATA_DIR, MODEL_DIR, EXP_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)


def run_episode(env, agent, max_n_steps, render=False):
    agent.reset()
    obs, info = env.reset()
    if render:
        env.render()
    log = []
    for _ in range(max_n_steps):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.reward(reward)
        log.append((obs, action, reward, next_obs, reward, terminated, truncated, info))
        obs = next_obs
        if render:
            env.render()
        if terminated or truncated:
            break
    return log


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def smooth(xs, win):
    return np.convolve(xs, np.ones(win) / win)


def get_env_dims(env):
    n_obs_dims = env.observation_space.low.size
    discrete = type(env.action_space) == gym.spaces.Discrete
    n_act_dims = env.action_space.n if discrete else env.action_space.low.size
    return n_obs_dims, n_act_dims, discrete
