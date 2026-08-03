"""
Convergence and Value Function Visualization Module.
Professional plotting utilities for research-quality figures.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom color palettes
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#73AB84',
    'warning': '#C73E1D',
    'background': '#F8F9FA',
    'text': '#2D3436'
}

def plot_value_function_heatmap(V, rows, cols, title="Value Function", 
                                save_path=None, cmap='viridis', figsize=(10, 8)):
    """
    Plot value function as a professional heatmap for grid environments.
    
    Parameters:
    -----------
    V : np.ndarray
        Value function array
    rows, cols : int
        Grid dimensions
    title : str
        Plot title
    save_path : str
        Path to save figure
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    """
    grid = V.reshape(rows, cols)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(grid, cmap=cmap, interpolation='bilinear', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add value annotations
    for i in range(rows):
        for j in range(cols):
            color = 'white' if grid[i, j] < np.mean(grid) else 'black'
            ax.text(j, i, f'{grid[i, j]:.2f}', 
                   ha='center', va='center', 
                   color=color, fontsize=9, fontweight='bold')
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    plt.close()


def plot_policy_arrows(Q, rows, cols, title="Learned Policy", 
                       save_path=None, figsize=(10, 8)):
    """
    Plot greedy policy as arrows on grid with professional styling.
    
    Parameters:
    -----------
    Q : np.ndarray
        Q-value table
    rows, cols : int
        Grid dimensions
    title : str
        Plot title
    save_path : str
        Path to save figure
    figsize : tuple
        Figure size
    """
    policy = np.argmax(Q, axis=1).reshape(rows, cols)
    
    # Action mapping: 0=up, 1=right, 2=down, 3=left
    arrows = {
        0: (0, -0.4),    # up
        1: (0.4, 0),     # right
        2: (0, 0.4),     # down
        3: (-0.4, 0)     # left
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw grid
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    for j in range(cols + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    
    # Draw policy arrows
    for i in range(rows):
        for j in range(cols):
            action = policy[i, j]
            dx, dy = arrows.get(action, (0, 0))
            
            # Draw arrow with professional styling
            ax.arrow(j, i, dx, dy, 
                    head_width=0.15, head_length=0.15,
                    fc=COLORS['primary'], ec=COLORS['primary'],
                    linewidth=2, alpha=0.8)
            
            # Add action label
            ax.text(j, i, str(action), 
                   ha='center', va='center',
                   fontsize=8, color='gray', alpha=0.5)
    
    # Styling
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    plt.close()


def plot_rmse_convergence(rmse_dict, title="RMSE Convergence", 
                          save_path=None, figsize=(12, 6)):
    """
    Plot RMSE convergence for multiple algorithms with professional styling.
    
    Parameters:
    -----------
    rmse_dict : dict
        Dictionary mapping algorithm names to RMSE lists
    title : str
        Plot title
    save_path : str
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use color palette
    colors = sns.color_palette("husl", len(rmse_dict))
    
    for idx, (name, rmse_list) in enumerate(rmse_dict.items()):
        episodes = np.arange(len(rmse_list))
        
        # Plot main line
        ax.plot(episodes, rmse_list, 
               label=name, linewidth=2.5,
               color=colors[idx])
        
        # Add confidence band (if multiple runs)
        if isinstance(rmse_list, np.ndarray) and rmse_list.ndim == 2:
            mean = np.mean(rmse_list, axis=0)
            std = np.std(rmse_list, axis=0)
            ax.fill_between(episodes, mean - std, mean + std,
                          alpha=0.2, color=colors[idx])
    
    # Styling
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Root Mean Square Error (RMSE)', fontsize=12, fontweight='bold')
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
