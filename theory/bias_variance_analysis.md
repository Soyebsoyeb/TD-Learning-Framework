# Bias-Variance Tradeoff Analysis

## The Bias-Variance Decomposition

For any estimator G of the true return:

```
MSE(G) = Bias(G)^2 + Var(G)
```

## Monte Carlo (n = infinity)

- **Bias**: 0 (unbiased estimator of the true return)
- **Variance**: High (depends on the entire trajectory)
- **Sample Efficiency**: Low (must wait until episode termination)

## TD(0) (n = 1)

- **Bias**: Non-zero (bootstraps with current value estimate)
- **Variance**: Low (depends only on one transition)
- **Sample Efficiency**: High (updates after every step)

## n-Step TD

- **Bias**: Decreases with n
- **Variance**: Increases with n
- **Tradeoff**: n controls the bias-variance tradeoff

## TD(lambda)

- **lambda = 0**: Equivalent to TD(0), high bias, low variance
- **lambda = 1**: Equivalent to Monte Carlo, zero bias, high variance
- **Intermediate lambda**: Balances bias and variance

## Empirical Observations

1. For small state spaces and short episodes, high lambda often performs best
2. For large state spaces or stochastic environments, moderate lambda (0.5-0.8) is typically optimal
3. True Online TD(lambda) often outperforms conventional TD(lambda) in early learning

## Control Algorithms

### SARSA vs Q-Learning

- **SARSA**: On-policy, considers exploration in updates, generally safer
- **Q-Learning**: Off-policy, directly optimizes greedy policy, can be riskier

### Cliff Walking Example

SARSA learns the safer path (slightly longer but avoids the cliff).
Q-learning learns the optimal path (shortest but risks falling off the cliff).
This demonstrates the on-policy vs off-policy behavior difference.

### Double Q-Learning

Reduces overestimation bias in Q-learning by approximately 50% in early training.
Particularly important in stochastic environments with noisy rewards.
