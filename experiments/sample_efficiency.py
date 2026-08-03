"""
Experiment 2: Sample Efficiency Comparison.
Measures reward per environment interaction.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.cliff_walking import CliffWalking
from algorithms.sarsa import SARSA
from algorithms.q_learning import QLearning
from algorithms.double_q_learning import DoubleQLearning
from algorithms.expected_sarsa import ExpectedSARSA


def run_sample_efficiency(n_episodes=500, n_runs=30):
    env_cls = CliffWalking
    env = env_cls()
    n_states = env.n_states
    n_actions = env.n_actions

    algorithms = {
        'SARSA': SARSA(n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1),
        'Q-Learning': QLearning(n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1),
        'Expected SARSA': ExpectedSARSA(n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1),
        'Double Q-Learning': DoubleQLearning(n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1),
    }

    results = {}
    for name, algo in algorithms.items():
        all_rewards = []
        for run in range(n_runs):
            algo.reset()
            env = env_cls()
            algo.train(env, n_episodes=n_episodes)
            window = 50
            smoothed = [np.mean(algo.episode_rewards[max(0, i-window):i+1]) for i in range(len(algo.episode_rewards))]
            all_rewards.append(smoothed)
        results[name] = np.array(all_rewards)
        print(f"{name}: Final avg reward = {np.mean(results[name][:, -1]):.2f}")

    return results


if __name__ == '__main__':
    results = run_sample_efficiency()
