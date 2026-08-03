"""
GridWorld Environment.
Configurable grid size, obstacles, terminal states.
"""
import numpy as np


class GridWorld:
    """
    Deterministic grid world.
    Actions: 0=up, 1=right, 2=down, 3=left.
    """
    def __init__(self, rows=5, cols=5, start=(0, 0), terminals=None, obstacles=None, step_reward=-1.0):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.terminals = set(terminals) if terminals else {(rows - 1, cols - 1)}
        self.obstacles = set(obstacles) if obstacles else set()
        self.step_reward = step_reward
        self.state = None
        self.n_states = rows * cols
        self.n_actions = 4
        self.reset()

    def _to_index(self, pos):
        return pos[0] * self.cols + pos[1]

    def _to_pos(self, idx):
        return (idx // self.cols, idx % self.cols)

    def reset(self):
        self.state = self.start
        return self._to_index(self.state), {}

    def step(self, action):
        r, c = self.state
        if action == 0:
            nr, nc = max(0, r - 1), c
        elif action == 1:
            nr, nc = r, min(self.cols - 1, c + 1)
        elif action == 2:
            nr, nc = min(self.rows - 1, r + 1), c
        elif action == 3:
            nr, nc = r, max(0, c - 1)
        else:
            raise ValueError("Invalid action")

        if (nr, nc) in self.obstacles:
            nr, nc = r, c

        self.state = (nr, nc)
        idx = self._to_index(self.state)
        reward = self.step_reward
        terminated = self.state in self.terminals
        if terminated:
            reward = 0.0
        return idx, reward, terminated, False, {}

    def get_transition_probs(self, state_idx, action):
        """For model-based methods."""
        self.state = self._to_pos(state_idx)
        next_idx, reward, terminated, _, _ = self.step(action)
        return {next_idx: (1.0, reward, terminated)}
