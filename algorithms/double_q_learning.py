"""
Double Q-Learning: Reduces maximization bias.
"""
import numpy as np


class DoubleQLearning:
    """
    Double Q-Learning maintains two independent Q estimators.
    """
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.QA = np.zeros((n_states, n_actions))
        self.QB = np.zeros((n_states, n_actions))
        self.episode_rewards = []
        self.episode_lengths = []

    def reset(self):
        self.QA = np.zeros((self.n_states, self.n_actions))
        self.QB = np.zeros((self.n_states, self.n_actions))
        self.episode_rewards = []
        self.episode_lengths = []

    def _reset_env(self, env):
        """Safely reset environment for different Gym versions."""
        result = env.reset()
        if isinstance(result, tuple):
            return result[0]
        return result

    def _step_env(self, env, action):
        """Safely step environment for different Gym versions."""
        result = env.step(action)
        if len(result) == 4:  # Old Gym
            next_state, reward, terminated, info = result
            return next_state, reward, terminated, False, info
        else:  # Gymnasium
            next_state, reward, terminated, truncated, info = result
            return next_state, reward, terminated, truncated, info

    def _select_action(self, state):
        Q_sum = self.QA[state] + self.QB[state]
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(Q_sum)

    def train(self, env, n_episodes=1000, max_steps=1000):
        for ep in range(n_episodes):
            state = self._reset_env(env)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            
            while not terminated and not truncated and steps < max_steps:
                action = self._select_action(state)
                next_state, reward, terminated, truncated, _ = self._step_env(env, action)
                
                if np.random.rand() < 0.5:
                    best_action = np.argmax(self.QA[next_state])
                    target = reward + (0.0 if terminated else self.gamma * self.QB[next_state, best_action])
                    td_error = target - self.QA[state, action]
                    self.QA[state, action] += self.alpha * td_error
                else:
                    best_action = np.argmax(self.QB[next_state])
                    target = reward + (0.0 if terminated else self.gamma * self.QA[next_state, best_action])
                    td_error = target - self.QB[state, action]
                    self.QB[state, action] += self.alpha * td_error
                
                state = next_state
                total_reward += reward
                steps += 1
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
        return (self.QA + self.QB) / 2.0

    def get_policy(self):
        return np.argmax(self.QA + self.QB, axis=1)
