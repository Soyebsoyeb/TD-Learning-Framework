# %%
"""
Algorithm Comparison Notebook
=============================
Compares control algorithms on Cliff Walking.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from environments.cliff_walking import CliffWalking
from algorithms.sarsa import SARSA
from algorithms.q_learning import QLearning
from algorithms.double_q_learning import DoubleQLearning
from algorithms.expected_sarsa import ExpectedSARSA

# %%
n_episodes = 500
n_runs = 30
env_info = CliffWalking()
n_states = env_info.n_states
n_actions = env_info.n_actions

algos = {
    'SARSA': SARSA(n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1),
    'Q-Learning': QLearning(n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1),
    'Expected SARSA': ExpectedSARSA(n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1),
    'Double Q-Learning': DoubleQLearning(n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1),
}

results = {}
window = 50

for name, algo_template in algos.items():
    all_rewards = []
    for run in range(n_runs):
        algo = type(algo_template)(n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1)
        if name == 'SARSA':
            algo = SARSA(n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1)
        elif name == 'Q-Learning':
            algo = QLearning(n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1)
        elif name == 'Expected SARSA':
            algo = ExpectedSARSA(n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1)
        elif name == 'Double Q-Learning':
            algo = DoubleQLearning(n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1)
        env = CliffWalking()
        algo.train(env, n_episodes=n_episodes)
        smoothed = [np.mean(algo.episode_rewards[max(0, i-window):i+1]) for i in range(len(algo.episode_rewards))]
        all_rewards.append(smoothed)
    results[name] = np.array(all_rewards)

# %%
plt.figure(figsize=(12, 7))
for name, data in results.items():
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    episodes = np.arange(len(mean))
    plt.plot(episodes, mean, label=name)
    plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)
plt.xlabel('Episode')
plt.ylabel('Average Return (smoothed)')
plt.title('Control Algorithm Comparison on Cliff Walking')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
