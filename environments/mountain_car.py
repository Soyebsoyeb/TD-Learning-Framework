"""
Mountain Car Environment with state discretization.
"""
import numpy as np


class MountainCar:
    """
    Continuous state mountain car discretized for tabular methods.
    """
    def __init__(self, position_bins=20, velocity_bins=20):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.position_bins = position_bins
        self.velocity_bins = velocity_bins
        self.n_actions = 3
        self.n_states = position_bins * velocity_bins
        self.position = None
        self.velocity = None
        self.reset()

    def reset(self):
        self.position = np.random.uniform(-0.6, -0.4)
        self.velocity = 0.0
        return self._get_state_index(), {}

    def _get_state_index(self):
        p_idx = int((self.position - self.min_position) / (self.max_position - self.min_position) * self.position_bins)
        v_idx = int((self.velocity + self.max_speed) / (2 * self.max_speed) * self.velocity_bins)
        p_idx = max(0, min(self.position_bins - 1, p_idx))
        v_idx = max(0, min(self.velocity_bins - 1, v_idx))
        return p_idx * self.velocity_bins + v_idx

    def step(self, action):
        force = (action - 1) * 0.001
        self.velocity += force + np.cos(3 * self.position) * (-0.0025)
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)
        self.position += self.velocity
        self.position = np.clip(self.position, self.min_position, self.max_position)
        if self.position == self.min_position and self.velocity < 0:
            self.velocity = 0.0

        reward = -1.0
        terminated = self.position >= self.goal_position
        return self._get_state_index(), reward, terminated, False, {}
