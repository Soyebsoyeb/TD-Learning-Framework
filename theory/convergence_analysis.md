# Convergence Analysis

## TD(0) Convergence

**Theorem**: For a finite MDP and lookup table representation, TD(0) with step sizes alpha_t satisfying Robbins-Monro conditions converges with probability 1 to V^pi.

**Proof Sketch**:
1. TD(0) is a stochastic approximation algorithm
2. The expected update is a contraction mapping
3. Apply the theory of stochastic approximation (Tsitsiklis & Van Roy, 1997)

## SARSA Convergence

**Theorem**: SARSA with epsilon-greedy exploration and decaying epsilon converges to the optimal policy under the GLIE (Greedy in the Limit with Infinite Exploration) assumption.

## Q-Learning Convergence

**Theorem**: Q-learning converges with probability 1 to Q* under the condition that all state-action pairs are visited infinitely often and step sizes satisfy Robbins-Monro conditions.

**Key Insight**: Q-learning is off-policy, so it does not require GLIE for convergence of the value function (though exploration is needed to find the optimal policy).

## Double Q-Learning Convergence

Double Q-learning converges to the same limit as Q-learning but with reduced variance in the early stages. Both estimators converge to Q*.

## Computational Complexity

| Algorithm | Per-Step Time | Per-Step Memory | Total Episodes |
|-----------|--------------|-----------------|----------------|
| TD(0)     | O(1)         | O(S)            | O(1/epsilon)   |
| n-Step TD | O(n)         | O(S + n)        | O(1/epsilon)   |
| TD(lambda)| O(S)         | O(S)            | O(1/epsilon)   |
| SARSA     | O(A)         | O(S*A)          | O(1/epsilon)   |
| Q-Learning| O(A)         | O(S*A)          | O(1/epsilon)   |
| Double Q  | O(A)         | O(2*S*A)        | O(1/epsilon)   |

Where S = number of states, A = number of actions.
