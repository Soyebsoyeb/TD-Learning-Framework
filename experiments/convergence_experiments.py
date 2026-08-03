"""
Experiment 3: Convergence experiments for lambda values.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.random_walk import RandomWalk
from algorithms.td_lambda import TDLambda
from algorithms.true_online_td_lambda import TrueOnlineTDLambda


def lambda_convergence(n_states=19, n_episodes=100, n_runs=100):
    lambdas = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    true_V = RandomWalk(n_states).true_values()
    results = {}

    for lam in lambdas:
        rmse_hist = []
        for run in range(n_runs):
            algo = TDLambda(n_states + 2, alpha=0.1, gamma=1.0, lambda_=lam)
            env = RandomWalk(n_states)
            algo.train(env, n_episodes=n_episodes)
            rmse = np.sqrt(np.mean((algo.V[1:n_states+1] - true_V[1:n_states+1])**2))
            rmse_hist.append(rmse)
        results[f'lambda={lam}'] = {
            'mean': np.mean(rmse_hist),
            'std': np.std(rmse_hist)
        }
        print(f"lambda={lam}: RMSE = {results[f'lambda={lam}']['mean']:.4f}")

    return results


if __name__ == '__main__':
    results = lambda_convergence()
