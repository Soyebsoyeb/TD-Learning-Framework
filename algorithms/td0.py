"""
TD(0) Algorithm.
State-value estimation with one-step bootstrapping.
"""
import numpy as np


class TD0:
    """
    Temporal Difference learning with n=1 step return.
    V(S_t) <- V(S_t) + alpha * [R_{t+1} + gamma * V(S_{t+1}) - V(S_t)]
    """
    def __init__(self, n_states, alpha=0.1, gamma=0.9):
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.V = np.zeros(n_states)
        self.episode_rewards = []
        self.episode_lengths = []

    def reset(self):
        self.V = np.zeros(self.n_states)
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
                
                # TD(0) update
                td_target = reward + (0.0 if terminated else self.gamma * self.V[next_state])
                td_error = td_target - self.V[state]
                self.V[state] += self.alpha * td_error
                
                state = next_state
                total_reward += reward
                steps += 1
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
        return self.V.copy()
