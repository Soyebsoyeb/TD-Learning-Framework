# %%
"""
TD(0) Analysis Notebook
=======================
Demonstrates TD(0) on Random Walk and compares with true values.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from environments.random_walk import RandomWalk
from algorithms.td0 import TD0

# %%
# Setup
n_states = 19
env = RandomWalk(n_states)
true_values = env.true_values()

# %%
# Train TD(0)
algo = TD0(n_states + 2, alpha=0.1, gamma=1.0)
V = algo.train(env, n_episodes=100)

# %%
# Plot comparison
states = np.arange(1, n_states + 1)
plt.figure(figsize=(10, 6))
plt.plot(states, true_values[1:n_states+1], 'o-', label='True Values')
plt.plot(states, V[1:n_states+1], 's-', label='TD(0) Estimate')
plt.xlabel('State')
plt.ylabel('Value')
plt.title('TD(0) Value Estimation vs True Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# RMSE over episodes
rmse_history = []
for ep in range(1, 101):
    algo.reset()
    env = RandomWalk(n_states)
    algo.train(env, n_episodes=ep)
    rmse = np.sqrt(np.mean((algo.V[1:n_states+1] - true_values[1:n_states+1])**2))
    rmse_history.append(rmse)

plt.figure(figsize=(10, 6))
plt.plot(rmse_history)
plt.xlabel('Episodes')
plt.ylabel('RMSE')
plt.title('TD(0) Convergence on 19-State Random Walk')
plt.grid(True, alpha=0.3)
plt.show()
