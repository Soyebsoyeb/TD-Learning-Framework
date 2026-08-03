"""
n-Step TD Algorithm.
Generalized n-step return for state-value estimation.
"""
import numpy as np


class NStepTD:
    """
    n-step TD prediction.
    G_{t:t+n} = R_{t+1} + gamma*R_{t+2} + ... + gamma^{n-1}*R_{t+n} + gamma^n * V(S_{t+n})
    """
    def __init__(self, n_states, n=3, alpha=0.1, gamma=0.9):
        self.n_states = n_states
        self.n = n
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
            states = [state]
            rewards = [0.0]
            T = float('inf')
            t = 0
            total_reward = 0.0
            terminated = False
            truncated = False
            steps = 0
            
            while True:
                if t < T:
                    # Select action
                    if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
                        action = np.random.randint(env.action_space.n)
                    elif hasattr(env, 'n_actions'):
                        action = np.random.randint(env.n_actions)
                    else:
                        action = np.random.choice([0, 1])
                    
                    next_state, reward, terminated, truncated, _ = self._step_env(env, action)
                    states.append(next_state)
                    rewards.append(reward)
                    total_reward += reward
                    steps += 1
                    if terminated or truncated:
                        T = t + 1
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0.0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        G += (self.gamma ** (i - tau - 1)) * rewards[i]
                    if tau + self.n < T:
                        G += (self.gamma ** self.n) * self.V[states[tau + self.n]]
                    s_tau = states[tau]
                    self.V[s_tau] += self.alpha * (G - self.V[s_tau])
                if tau == T - 1:
                    break
                t += 1
                if t > max_steps:
                    break
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
        return self.V.copy()
