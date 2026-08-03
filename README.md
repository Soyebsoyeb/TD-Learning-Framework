# Unified Temporal Difference Learning Framework

## From Classical TD Methods to Modern Value-Based Reinforcement Learning

A complete, research-grade implementation of Temporal Difference (TD) learning algorithms from scratch in Python. This project demonstrates deep understanding of TD learning theory, algorithms, empirical evaluation, and visualization.

## Project Structure

```
TD-Learning-Framework/
|-- algorithms/          # Core algorithm implementations
|   |-- td0.py
|   |-- n_step_td.py
|   |-- td_lambda.py
|   |-- true_online_td_lambda.py
|   |-- sarsa.py
|   |-- expected_sarsa.py
|   |-- q_learning.py
|   |-- double_q_learning.py
|   |-- watkins_q_lambda.py
|
|-- environments/        # Custom environments
|   |-- random_walk.py
|   |-- gridworld.py
|   |-- cliff_walking.py
|   |-- windy_gridworld.py
|   |-- mountain_car.py
|   |-- cartpole_wrapper.py
|
|-- theory/              # Mathematical documentation
|   |-- td_derivations.md
|   |-- convergence_analysis.md
|   |-- bias_variance_analysis.md
|
|-- experiments/         # Benchmark experiments
|   |-- compare_algorithms.py
|   |-- sample_efficiency.py
|   |-- convergence_experiments.py
|   |-- runtime_benchmark.py
|
|-- visualization/       # Plotting and dashboard
|   |-- learning_curves.py
|   |-- convergence_plots.py
|   |-- dashboard.py
|
|-- notebooks/           # Analysis notebooks
|   |-- TD0_analysis.ipynb.py
|   |-- Lambda_methods.ipynb.py
|   |-- Algorithm_comparison.ipynb.py
|
|-- tests/               # Unit tests
|-- README.md
|-- requirements.txt
|-- LICENSE
```

## Implemented Algorithms

### Prediction (State-Value Estimation)
- **TD(0)**: One-step bootstrapping
- **n-Step TD**: Generalized n-step returns
- **TD(lambda)**: Eligibility traces with accumulating traces
- **True Online TD(lambda)**: Dutch traces (van Seijen & Sutton, 2014)

### Control (Action-Value Estimation)
- **SARSA**: On-policy TD control
- **Expected SARSA**: Uses expectation over policy
- **Q-Learning**: Off-policy TD control
- **Double Q-Learning**: Reduces maximization bias
- **Watkins Q(lambda)**: Q-learning with eligibility traces and trace cutting

## Environments

All environments implemented from scratch:
- **Random Walk**: 5-state and 19-state versions with analytical true values
- **GridWorld**: Configurable size, obstacles, terminal states
- **Cliff Walking**: Classic SARSA vs Q-learning demonstration
- **Windy GridWorld**: Stochastic wind transitions
- **Mountain Car**: Continuous state with discretization
- **CartPole**: Gymnasium wrapper with tabular discretization

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Experiments

```bash
# Compare prediction algorithms
python experiments/compare_algorithms.py

# Sample efficiency on Cliff Walking
python experiments/sample_efficiency.py

# Lambda convergence analysis
python experiments/convergence_experiments.py

# Runtime benchmark
python experiments/runtime_benchmark.py
```

### Interactive Dashboard

```bash
streamlit run visualization/dashboard.py
```

### Run Tests

```bash
python -m pytest tests/
```

## Mathematical Theory

See `theory/` directory for detailed derivations:
- Bellman equation foundations
- TD error as stochastic gradient descent
- Eligibility trace derivations
- Convergence proofs and conditions
- Bias-variance tradeoff analysis

## Key References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD Thesis, Cambridge University.
- Sutton, R. S. (1988). Learning to Predict by the Methods of Temporal Differences. *Machine Learning*, 3(1), 9-44.
- van Seijen, H., & Sutton, R. S. (2014). True Online Temporal-Difference Learning. *Journal of Machine Learning Research*, 17(145), 1-40.

## Requirements

- Python 3.11+
- NumPy
- PyTorch (optional, for neural approximators)
- Matplotlib
- Streamlit
- Gymnasium
- pytest

## License

MIT License
