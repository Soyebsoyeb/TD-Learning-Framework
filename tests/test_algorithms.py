"""
Unit tests for TD learning algorithms.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.random_walk import RandomWalk
from environments.gridworld import GridWorld
from environments.cliff_walking import CliffWalking
from algorithms.td0 import TD0
from algorithms.td_lambda import TDLambda
from algorithms.sarsa import SARSA
from algorithms.q_learning import QLearning
from algorithms.double_q_learning import DoubleQLearning


def test_random_walk_true_values():
    env = RandomWalk(5)
    true = env.true_values()
    assert true[0] == 0.0
    assert true[6] == 1.0
    assert true[3] == 0.5


def test_td0_convergence():
    env = RandomWalk(5)
    algo = TD0(7, alpha=0.1, gamma=1.0)
    V = algo.train(env, n_episodes=500)
    true = env.true_values()
    rmse = np.sqrt(np.mean((V[1:6] - true[1:6])**2))
    assert rmse < 0.1, f"TD(0) did not converge: RMSE={rmse}"


def test_td_lambda_shape():
    env = RandomWalk(5)
    algo = TDLambda(7, alpha=0.1, gamma=1.0, lambda_=0.9)
    V = algo.train(env, n_episodes=100)
    assert len(V) == 7


def test_sarsa_q_shape():
    env = CliffWalking()
    algo = SARSA(env.n_states, env.n_actions, alpha=0.1, gamma=1.0, epsilon=0.1)
    Q = algo.train(env, n_episodes=100)
    assert Q.shape == (env.n_states, env.n_actions)


def test_q_learning_vs_sarsa():
    env = CliffWalking()
    sarsa = SARSA(env.n_states, env.n_actions, alpha=0.5, gamma=1.0, epsilon=0.1)
    qlearn = QLearning(env.n_states, env.n_actions, alpha=0.5, gamma=1.0, epsilon=0.1)
    sarsa.train(env, n_episodes=200)
    qlearn.train(CliffWalking(), n_episodes=200)
    assert sarsa.Q.shape == qlearn.Q.shape


def test_double_q_learning():
    env = CliffWalking()
    algo = DoubleQLearning(env.n_states, env.n_actions, alpha=0.5, gamma=1.0, epsilon=0.1)
    Q = algo.train(env, n_episodes=200)
    assert Q.shape == (env.n_states, env.n_actions)


def test_gridworld_step():
    env = GridWorld(5, 5)
    state, _ = env.reset()
    assert 0 <= state < 25
    next_state, reward, terminated, _, _ = env.step(1)
    assert 0 <= next_state < 25


def test_cliff_walking_terminal():
    env = CliffWalking()
    env.state = (3, 10)
    next_state, reward, terminated, _, _ = env.step(1)
    assert terminated is True


if __name__ == '__main__':
    test_random_walk_true_values()
    test_td0_convergence()
    test_td_lambda_shape()
    test_sarsa_q_shape()
    test_q_learning_vs_sarsa()
    test_double_q_learning()
    test_gridworld_step()
    test_cliff_walking_terminal()
    print("All tests passed.")
