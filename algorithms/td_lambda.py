"""
TD(lambda) Algorithm with accumulating eligibility traces.
"""
import numpy as np


class TDLambda:
    """
    TD(lambda) with accumulating traces.
    e_t(s) = gamma * lambda * e_{t-1}(s) + 1(S_t = s)
    """
    def __init__(self, n_states, alpha=0.1, gamma=0.9, lambda_=0.9):
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.V = np.zeros(n_states)
        self.E = np.zeros(n_states)
        self.episode_rewards = []
        self.episode_lengths = []

    def reset(self):
        self.V = np.zeros(self.n_states)
        self.E = np.zeros(self.n_states)
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

    def train(self, env, n_episodes=1000, max_steps=1000):
        for ep in range(n_episodes):
            state = self._reset_env(env)
            self.E.fill(0.0)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            
            while not terminated and not truncated and steps < max_steps:
                # Select action
                if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
                    action = np.random.randint(env.action_space.n)
                elif hasattr(env, 'n_actions'):
                    action = np.random.randint(env.n_actions)
                else:
                    action = np.random.choice([0, 1])
                
                next_state, reward, terminated, truncated, _ = self._step_env(env, action)
                delta = reward + (0.0 if terminated else self.gamma * self.V[next_state]) - self.V[state]
                self.E[state] += 1.0
                self.V += self.alpha * delta * self.E
                self.E *= self.gamma * self.lambda_
                state = next_state
                total_reward += reward
                steps += 1
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
        return self.V.copy()
