# %%
"""
Lambda Methods Analysis Notebook
=================================
Compares TD(lambda) and True Online TD(lambda) across different lambda values.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from environments.random_walk import RandomWalk
from algorithms.td_lambda import TDLambda
from algorithms.true_online_td_lambda import TrueOnlineTDLambda

# %%
n_states = 19
n_episodes = 100
n_runs = 50
lambdas = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
true_values = RandomWalk(n_states).true_values()

results_td = {}
results_true = {}

for lam in lambdas:
    rmse_td = []
    rmse_true = []
    for run in range(n_runs):
        algo1 = TDLambda(n_states + 2, alpha=0.1, gamma=1.0, lambda_=lam)
        algo2 = TrueOnlineTDLambda(n_states + 2, alpha=0.1, gamma=1.0, lambda_=lam)
        env1 = RandomWalk(n_states)
        env2 = RandomWalk(n_states)
        algo1.train(env1, n_episodes=n_episodes)
        algo2.train(env2, n_episodes=n_episodes)
        rmse_td.append(np.sqrt(np.mean((algo1.V[1:n_states+1] - true_values[1:n_states+1])**2)))
        rmse_true.append(np.sqrt(np.mean((algo2.V[1:n_states+1] - true_values[1:n_states+1])**2)))
    results_td[lam] = (np.mean(rmse_td), np.std(rmse_td))
    results_true[lam] = (np.mean(rmse_true), np.std(rmse_true))

# %%
plt.figure(figsize=(10, 6))
means_td = [results_td[lam][0] for lam in lambdas]
stds_td = [results_td[lam][1] for lam in lambdas]
means_true = [results_true[lam][0] for lam in lambdas]
stds_true = [results_true[lam][1] for lam in lambdas]

plt.errorbar(lambdas, means_td, yerr=stds_td, fmt='o-', label='TD(lambda)', capsize=5)
plt.errorbar(lambdas, means_true, yerr=stds_true, fmt='s-', label='True Online TD(lambda)', capsize=5)
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title('Effect of Lambda on Prediction Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
