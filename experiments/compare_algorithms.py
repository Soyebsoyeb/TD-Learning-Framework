"""
Experiment 1: Compare TD(0) vs Monte Carlo and other algorithms.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.random_walk import RandomWalk
from algorithms.td0 import TD0
from algorithms.n_step_td import NStepTD
from algorithms.td_lambda import TDLambda
from algorithms.true_online_td_lambda import TrueOnlineTDLambda


def run_comparison(n_states=19, n_episodes=200, n_runs=50):
    results = {}
    true_V = RandomWalk(n_states).true_values()

    algorithms = {
        'TD(0)': TD0(n_states + 2, alpha=0.1, gamma=1.0),
        'n-Step TD (n=3)': NStepTD(n_states + 2, n=3, alpha=0.1, gamma=1.0),
        'TD(lambda=0.5)': TDLambda(n_states + 2, alpha=0.1, gamma=1.0, lambda_=0.5),
        'TD(lambda=0.9)': TDLambda(n_states + 2, alpha=0.1, gamma=1.0, lambda_=0.9),
        'True Online TD(lambda=0.9)': TrueOnlineTDLambda(n_states + 2, alpha=0.1, gamma=1.0, lambda_=0.9),
    }

    for name, algo in algorithms.items():
        rmse_history = []
        for run in range(n_runs):
            algo.reset()
            env = RandomWalk(n_states)
            algo.train(env, n_episodes=n_episodes)
            rmse = np.sqrt(np.mean((algo.V[1:n_states+1] - true_V[1:n_states+1])**2))
            rmse_history.append(rmse)
        results[name] = {
            'mean_rmse': np.mean(rmse_history),
            'std_rmse': np.std(rmse_history),
            'rewards': algo.episode_rewards
        }
        print(f"{name}: RMSE = {results[name]['mean_rmse']:.4f} +/- {results[name]['std_rmse']:.4f}")

    return results


if __name__ == '__main__':
    results = run_comparison()
