"""
Random Walk Environment.
Implements 5-state and 19-state random walk.
"""
import numpy as np


class RandomWalk:
    """
    Random walk on a linear chain of states.
    States are numbered 1 to n_states.
    State 0 and n_states+1 are terminal (left and right).
    """
    def __init__(self, n_states=5):
        self.n_states = n_states
        self.state = None
        self.terminal_left = 0
        self.terminal_right = n_states + 1
        self.reset()

    def reset(self):
        self.state = (self.n_states + 1) // 2
        return self.state

    def step(self, action=None):
        """
        Action is ignored; transitions are random.
        Returns: next_state, reward, terminated, truncated, info
        """
        direction = np.random.choice([-1, 1])
        next_state = self.state + direction
        reward = 0.0
        terminated = False
        if next_state == self.terminal_right:
            reward = 1.0
            terminated = True
        elif next_state == self.terminal_left:
            reward = 0.0
            terminated = True
        self.state = next_state
        return next_state, reward, terminated, False, {}

    def true_values(self):
        """
        Analytical state values for random walk.
        V(s) = s / (n_states + 1) for right terminal reward 1.
        """
        values = np.zeros(self.n_states + 2)
        for s in range(1, self.n_states + 1):
            values[s] = s / (self.n_states + 1)
        return values
