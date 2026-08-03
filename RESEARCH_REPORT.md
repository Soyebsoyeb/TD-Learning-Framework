# Research Report: Unified Temporal Difference Learning Framework

## Abstract

This report presents a complete implementation and empirical analysis of classical and modern Temporal Difference (TD) learning algorithms. We implement nine algorithms from scratch and evaluate them across six custom environments. Our experiments confirm theoretical predictions regarding convergence, bias-variance tradeoffs, and the benefits of double learning and true online updates.

## 1. Introduction

Temporal Difference learning is the foundation of modern reinforcement learning. This project implements a unified framework covering prediction algorithms (TD(0), n-step TD, TD(lambda), True Online TD(lambda)) and control algorithms (SARSA, Expected SARSA, Q-Learning, Double Q-Learning, Watkins Q(lambda)).

## 2. Methods

### 2.1 Algorithms

All algorithms are implemented using only NumPy for numerical computation, with PyTorch reserved for potential neural approximator extensions. No existing RL libraries (Stable-Baselines3, RLlib, CleanRL) are used.

### 2.2 Environments

Six environments are implemented from scratch:
1. Random Walk (5-state and 19-state)
2. GridWorld (configurable)
3. Cliff Walking
4. Windy GridWorld
5. Mountain Car (discretized)
6. CartPole (Gymnasium wrapper with discretization)

### 2.3 Experimental Protocol

Each experiment uses controlled random seeds and is averaged over multiple independent runs. Learning curves are smoothed with a moving average window of 50 episodes.

## 3. Results

### 3.1 Prediction Accuracy

On the 19-state random walk, True Online TD(lambda=0.9) achieves the lowest RMSE after 100 episodes, followed closely by conventional TD(lambda=0.9). TD(0) exhibits higher initial bias but converges reliably.

### 3.2 Bias-Variance Tradeoff

As predicted by theory, lambda=0 (TD(0)) shows high bias and low variance, while lambda=1 (Monte Carlo equivalent) shows zero bias but high variance. Optimal performance is achieved at lambda=0.7-0.9.

### 3.3 Control Algorithm Comparison

On Cliff Walking:
- **SARSA** learns the safer path around the cliff
- **Q-Learning** learns the optimal but riskier path along the cliff edge
- **Expected SARSA** converges faster than SARSA due to lower variance updates
- **Double Q-Learning** shows reduced overestimation in early episodes

### 3.4 Runtime Performance

All tabular algorithms exhibit O(S*A) per-step complexity. TD(0) is fastest per step. Watkins Q(lambda) incurs overhead from trace maintenance.

## 4. Discussion

The empirical results align closely with theoretical predictions from Sutton & Barto (2018). Key findings:

1. True Online TD(lambda) provides marginal but consistent improvements over conventional TD(lambda)
2. Double Q-Learning effectively reduces maximization bias in stochastic settings
3. Expected SARSA offers a favorable bias-variance tradeoff for on-policy control
4. The choice of lambda significantly impacts sample efficiency

## 5. Conclusion

This framework provides a complete, reproducible implementation of core TD learning methods suitable for research and education. All code, experiments, and visualizations are available in the accompanying repository.

## References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
2. Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. PhD Thesis.
3. van Seijen, H., & Sutton, R. S. (2014). True Online Temporal-Difference Learning. JMLR.
4. Hasselt, H. V. (2010). Double Q-Learning. NeurIPS.
