"""
CartPole wrapper using Gymnasium interface but manual algorithm implementation.
"""
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym


class CartPoleTabular:
    """
    Wraps Gymnasium CartPole with state discretization for tabular RL.
    """
    def __init__(self, bins=(10, 10, 10, 10)):
        self.env = gym.make("CartPole-v1")
        self.bins = bins
        self.n_actions = self.env.action_space.n
        self.n_states = np.prod(bins)
        self.state = None
        self._state_ranges = [
            (-4.8, 4.8),
            (-4.0, 4.0),
            (-0.418, 0.418),
            (-4.0, 4.0)
        ]

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.state = self._discretize(obs)
        return self.state, info

    def _discretize(self, obs):
        indices = []
        for i, (val, (low, high)) in enumerate(zip(obs, self._state_ranges)):
            val = np.clip(val, low, high)
            idx = int((val - low) / (high - low) * self.bins[i])
            idx = min(idx, self.bins[i] - 1)
            indices.append(idx)
        state_idx = 0
        for i, idx in enumerate(indices):
            state_idx = state_idx * self.bins[i] + idx
        return state_idx

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.state = self._discretize(obs)
        return self.state, reward, terminated, truncated, info

    def close(self):
        self.env.close()
