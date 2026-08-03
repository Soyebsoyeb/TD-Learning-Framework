"""
Learning Curve Visualization Module.
Professional plotting utilities for training dynamics analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_learning_curves(results, title="Learning Curves", 
                         save_path=None, figsize=(12, 6),
                         show_std=True, smooth_window=None):
    """
    Plot learning curves for multiple algorithms with professional styling.
    
    Parameters:
    -----------
    results : dict
        Dictionary mapping algorithm names to arrays of shape (n_runs, n_episodes)
    title : str
        Plot title
    save_path : str
        Path to save figure
    figsize : tuple
        Figure size
    show_std : bool
        Whether to show standard deviation bands
    smooth_window : int
        Window size for smoothing (None for no smoothing)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use color palette
    colors = sns.color_palette("husl", len(results))
    
    for idx, (name, data) in enumerate(results.items()):
        # Calculate statistics
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        episodes = np.arange(len(mean))
        
        # Apply smoothing if requested
        if smooth_window:
            mean = gaussian_filter1d(mean, sigma=smooth_window/10)
        
        # Plot main line
        ax.plot(episodes, mean, 
               label=name, linewidth=2.5,
               color=colors[idx])
        
        # Add standard deviation band
        if show_std:
            ax.fill_between(episodes, mean - std, mean + std,
                          alpha=0.2, color=colors[idx])
    
    # Styling
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Return', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    plt.close()


def plot_smoothed_rewards(reward_dict, window=50, title="Smoothed Rewards",
                          save_path=None, figsize=(12, 6)):
    """
    Plot smoothed reward curves for multiple algorithms.
    
    Parameters:
    -----------
    reward_dict : dict
        Dictionary mapping algorithm names to reward lists
    window : int
        Smoothing window size
    title : str
        Plot title
    save_path : str
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use color palette
    colors = sns.color_palette("husl", len(reward_dict))
    
    for idx, (name, rewards) in enumerate(reward_dict.items()):
        # Calculate smoothed rewards
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        episodes = np.arange(window-1, len(rewards))
        
        # Plot smoothed line
        ax.plot(episodes, smoothed, 
               label=name, linewidth=2.5,
               color=colors[idx])
        
        # Add raw rewards as faint background
        ax.plot(rewards, alpha=0.1, color=colors[idx], linewidth=0.5)
    
    # Styling
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Average Return (window={window})', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    plt.close()


def plot_comparison_dashboard(results, title="Algorithm Comparison",
                              save_path=None, figsize=(14, 10)):
    """
    Create a comprehensive comparison dashboard with multiple subplots.
    
    Parameters:
    -----------
    results : dict
        Dictionary mapping algorithm names to arrays of shape (n_runs, n_episodes)
    title : str
        Overall title
    save_path : str
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig = plt.figure(figsize=figsize)
    
    # Create subplot grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Learning curves
    ax1 = fig.add_subplot(gs[0, :])
    colors = sns.color_palette("husl", len(results))
    
    for idx, (name, data) in enumerate(results.items()):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        episodes = np.arange(len(mean))
        
        ax1.plot(episodes, mean, label=name, linewidth=2.5, color=colors[idx])
        ax1.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=colors[idx])
    
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Average Return', fontsize=11)
    ax1.set_title('Learning Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Final performance comparison
    ax2 = fig.add_subplot(gs[1, 0])
    final_performance = []
    algo_names = []
    
    for name, data in results.items():
        final_performance.append(np.mean(data[:, -50:]))
        algo_names.append(name)
    
    bars = ax2.bar(algo_names, final_performance, color=colors)
    ax2.set_ylabel('Average Return (Last 50)', fontsize=11)
    ax2.set_title('Final Performance', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_performance):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 3: Convergence speed
    ax3 = fig.add_subplot(gs[1, 1])
    convergence_speed = []
    
    for name, data in results.items():
        mean = np.mean(data, axis=0)
        # Find episode where performance reaches 90% of final
        final_perf = np.mean(mean[-50:])
        threshold = 0.9 * final_perf
        idx = np.where(mean >= threshold)[0]
        speed = idx[0] if len(idx) > 0 else len(mean)
        convergence_speed.append(speed)
    
    bars = ax3.bar(algo_names, convergence_speed, color=colors)
    ax3.set_ylabel('Episodes to Converge', fontsize=11)
    ax3.set_title('Convergence Speed (90% of final)', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, convergence_speed):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value}', ha='center', va='bottom', fontsize=9)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    plt.close()
