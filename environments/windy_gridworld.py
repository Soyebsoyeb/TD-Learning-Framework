"""
Windy GridWorld Environment.
Upward wind shifts the agent stochastically.
"""
import numpy as np


class WindyGridWorld:
    """
    7x10 grid with upward wind in certain columns.
    """
    def __init__(self, rows=7, cols=10, wind=None, start=(3, 0), goal=(3, 7)):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        if wind is None:
            self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        else:
            self.wind = wind
        self.n_actions = 4
        self.n_states = rows * cols
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
            nr, nc = r - 1, c
        elif action == 1:
            nr, nc = r, c + 1
        elif action == 2:
            nr, nc = r + 1, c
        elif action == 3:
            nr, nc = r, c - 1
        else:
            raise ValueError("Invalid action")

        wind = self.wind[c] if 0 <= c < self.cols else 0
        wind_shift = np.random.choice([wind, wind + 1, wind - 1], p=[1/3, 1/3, 1/3])
        nr -= wind_shift
        nr = max(0, min(self.rows - 1, nr))
        nc = max(0, min(self.cols - 1, nc))

        self.state = (int(nr), int(nc))
        reward = -1.0
        terminated = self.state == self.goal
        return self._to_index(self.state), reward, terminated, False, {}
