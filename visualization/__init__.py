"""
Visualization Module for TD Learning Framework.

This module provides comprehensive visualization tools for analyzing
Temporal Difference Learning algorithms, including:
- Learning curves and convergence plots
- Value function heatmaps
- Policy visualization
- Interactive dashboards
"""

from .learning_curves import plot_learning_curves, plot_smoothed_rewards, plot_comparison_dashboard
from .convergence_plots import (
    plot_value_function_heatmap,
    plot_policy_arrows,
    plot_rmse_convergence
)

__all__ = [
    'plot_learning_curves',
    'plot_smoothed_rewards',
    'plot_comparison_dashboard',
    'plot_value_function_heatmap',
    'plot_policy_arrows',
    'plot_rmse_convergence'
]

__version__ = '2.0.0'
__author__ = 'TD Learning Research Group'
