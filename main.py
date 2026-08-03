"""
Main runner script for the TD Learning Framework.
Executes all experiments and generates visualizations.
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from experiments.compare_algorithms import run_comparison
from experiments.sample_efficiency import run_sample_efficiency
from experiments.convergence_experiments import lambda_convergence
from experiments.runtime_benchmark import benchmark

from visualization.learning_curves import plot_learning_curves, plot_smoothed_rewards
from visualization.convergence_plots import plot_value_function_heatmap, plot_rmse_convergence


def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("Unified Temporal Difference Learning Framework")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)

    print("\n[1/4] Running algorithm comparison on Random Walk...")
    comp_results = run_comparison(n_states=19, n_episodes=200, n_runs=50)
    print("Done.\n")

    print("[2/4] Running sample efficiency experiment on Cliff Walking...")
    sample_results = run_sample_efficiency(n_episodes=500, n_runs=30)
    plot_learning_curves(sample_results, title="Sample Efficiency: Cliff Walking", save_path="results/sample_efficiency.png")
    print("Done. Saved to results/sample_efficiency.png\n")

    print("[3/4] Running lambda convergence experiment...")
    lambda_results = lambda_convergence(n_states=19, n_episodes=100, n_runs=100)
    print("Done.\n")

    print("[4/4] Running runtime benchmark...")
    bench_results = benchmark()
    print("Done.\n")

    print("=" * 60)
    print("All experiments completed. Results saved to results/")
    print("=" * 60)


if __name__ == '__main__':
    main()
