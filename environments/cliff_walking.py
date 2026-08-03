"""
Cliff Walking Environment.
Classic grid where falling off the cliff resets to start with large negative reward.
"""
import numpy as np


class CliffWalking:
    """
    4x12 grid. Start at bottom-left, goal at bottom-right.
    Cliff is the bottom row except start and goal.
    """
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.n_actions = 4
        self.n_states = self.rows * self.cols
        self.state = None
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

        if nr == 3 and 1 <= nc <= 10:
            reward = -100.0
            self.state = self.start
            terminated = False
        else:
            reward = -1.0
            self.state = (nr, nc)
            terminated = self.state == self.goal

        return self._to_index(self.state), reward, terminated, False, {}
