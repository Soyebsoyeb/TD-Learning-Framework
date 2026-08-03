# Temporal Difference Learning: Mathematical Derivations

## 1. TD(0) Derivation

### Bellman Equation for State Values

For a Markov Decision Process (MDP) with policy pi, the state-value function satisfies:

```
V^pi(s) = E_pi[R_{t+1} + gamma * V^pi(S_{t+1}) | S_t = s]
```

### TD Error as Bellman Residual

The Bellman error for an estimate V is:

```
delta_t = R_{t+1} + gamma * V(S_{t+1}) - V(S_t)
```

This is the one-step temporal difference error. TD(0) performs stochastic gradient descent on the squared Bellman error:

```
V(S_t) <- V(S_t) + alpha * delta_t
```

### Convergence

Under standard Robbins-Monro conditions on the step size alpha:
- sum(alpha_t) = infinity
- sum(alpha_t^2) < infinity

TD(0) converges with probability 1 to V^pi for lookup table representations.

## 2. n-Step Return

The n-step return bootstraps after n steps:

```
G_{t:t+n} = sum_{i=1}^{n} gamma^{i-1} * R_{t+i} + gamma^n * V(S_{t+n})
```

Special cases:
- n = 1: TD(0)
- n = infinity: Monte Carlo

## 3. TD(lambda) with Eligibility Traces

### Forward View (lambda-return)

```
G_t^lambda = (1 - lambda) * sum_{n=1}^{infinity} lambda^{n-1} * G_{t:t+n}
```

### Backward View (Eligibility Traces)

Accumulating traces:
```
e_t(s) = gamma * lambda * e_{t-1}(s) + 1(S_t = s)
```

Update:
```
V(s) <- V(s) + alpha * delta_t * e_t(s)   for all s
```

### Equivalence

For linear function approximation and lookup tables, the forward and backward views are equivalent.

## 4. True Online TD(lambda)

Dutch traces provide exact equivalence to the forward view:

```
e_t = gamma * lambda * e_{t-1} + alpha * (1 - gamma * lambda * e_{t-1}^T * x_t) * x_t
```

This eliminates the discrepancy between online and offline updates.

## 5. SARSA Derivation

SARSA is on-policy control. The action-value Bellman equation:

```
Q^pi(s,a) = E[R_{t+1} + gamma * Q^pi(S_{t+1}, A_{t+1}) | S_t=s, A_t=a]
```

Update:
```
Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [R_{t+1} + gamma * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

## 6. Q-Learning Derivation

Q-learning is off-policy. It uses the maximum over next actions:

```
Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [R_{t+1} + gamma * max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
```

This directly approximates the optimal action-value function Q*.

## 7. Double Q-Learning

Maximization bias occurs because max_a Q(s,a) is a biased estimator of max_a Q*(s,a).

Double Q-learning maintains two estimators QA and QB:
- Update QA using QB to evaluate the maximizing action
- Update QB using QA to evaluate the maximizing action

This decouples action selection from action evaluation, reducing overestimation.
