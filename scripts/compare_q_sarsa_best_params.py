#!/usr/bin/env python3
"""
compare_q_sarsa_best_params.py

Script to compare Q-learning and SARSA using the best-performing hyperparameters.
Generates comparative visualizations and summary of differences.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from typing import Dict, Tuple, Optional
import seaborn as sns

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.model import Model, Actions
from environment.world_config import small_world
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SarsaAgent
from utils.plot_vp import plot_vp


def run_comparison(model: Model, hyperparams: Dict):
    """
    Run Q-learning and SARSA with the best hyperparameters and compare results.
    
    Args:
        model: The environment model
        hyperparams: Dictionary of best hyperparameters
    
    Returns:
        q_learning_results: Results from Q-learning
        sarsa_results: Results from SARSA
    """
    # Set the model's gamma value
    original_gamma = model.gamma
    model.gamma = hyperparams['gamma']
    
    print(f"Running comparison with best hyperparameters:")
    print(f"  α = {hyperparams['alpha']}")
    print(f"  ε = {hyperparams['epsilon']}")
    print(f"  γ = {hyperparams['gamma']}")
    print(f"  Episodes = {hyperparams['num_episodes']}")
    print(f"  Max steps = {hyperparams['max_steps']}")
    
    # Create and train Q-learning agent
    print("\nTraining Q-learning agent...")
    q_agent = QLearningAgent(
        model=model,
        alpha=hyperparams['alpha'],
        epsilon=hyperparams['epsilon']
    )
    
    q_start = time.time()
    q_policy, q_values, q_stats = q_agent.train(
        num_episodes=hyperparams['num_episodes'],
        max_steps=hyperparams['max_steps'],
        verbose=True
    )
    q_time = time.time() - q_start
    
    # Create and train SARSA agent
    print("\nTraining SARSA agent...")
    sarsa_agent = SarsaAgent(
        model=model,
        alpha=hyperparams['alpha'],
        epsilon=hyperparams['epsilon']
    )
    
    sarsa_start = time.time()
    sarsa_policy, sarsa_values, sarsa_stats = sarsa_agent.train(
        num_episodes=hyperparams['num_episodes'],
        max_steps=hyperparams['max_steps'],
        expected_sarsa=False,
        verbose=True
    )
    sarsa_time = time.time() - sarsa_start
    
    # Calculate value functions from Q-values
    q_V = np.max(q_values, axis=1)
    sarsa_V = np.max(sarsa_values, axis=1)
    
    # Package results
    q_learning_results = {
        'policy': q_policy,
        'Q': q_values,
        'V': q_V,
        'stats': q_stats,
        'training_time': q_time
    }
    
    sarsa_results = {
        'policy': sarsa_policy,
        'Q': sarsa_values,
        'V': sarsa_V,
        'stats': sarsa_stats,
        'training_time': sarsa_time
    }
    
    # Plot learning curves
    plot_learning_curves(q_learning_results, sarsa_results)
    
    # Plot steps per episode
    plot_steps_per_episode(q_learning_results, sarsa_results)
    
    # Visualize policies and value functions
    diff_states = visualize_policies_and_values(model, q_learning_results, sarsa_results)
    
    # Generate Q-value heatmaps
    plot_q_value_heatmaps(model, "Q-learning", q_values)
    plot_q_value_heatmaps(model, "SARSA", sarsa_values)
    
    # Generate and save summary
    generate_summary(model, q_learning_results, sarsa_results)
    
    # Restore original gamma
    model.gamma = original_gamma
    
    return q_learning_results, sarsa_results


def plot_learning_curves(q_learning_results: Dict, sarsa_results: Dict):
    """
    Plot comparative learning curves (episode returns).
    
    Args:
        q_learning_results: Results from Q-learning
        sarsa_results: Results from SARSA
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Q-learning vs SARSA: Learning Curves', fontsize=16)
    
    # Raw episode returns
    q_returns = q_learning_results['stats']['episode_returns']
    sarsa_returns = sarsa_results['stats']['episode_returns']
    
    ax1.plot(q_returns, label='Q-learning', alpha=0.7)
    ax1.plot(sarsa_returns, label='SARSA', alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title('Episode Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Smoothed episode returns
    window_size = min(50, len(q_returns))
    
    smoothed_q = np.convolve(q_returns, np.ones(window_size)/window_size, mode='valid')
    smoothed_sarsa = np.convolve(sarsa_returns, np.ones(window_size)/window_size, mode='valid')
    
    ax2.plot(smoothed_q, label='Q-learning', linewidth=2)
    ax2.plot(smoothed_sarsa, label='SARSA', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Smoothed Return')
    ax2.set_title(f'Smoothed Episode Returns (window={window_size})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('q_vs_sarsa_learning_curves.png', dpi=300, bbox_inches='tight')


def plot_steps_per_episode(q_learning_results: Dict, sarsa_results: Dict):
    """
    Plot comparative steps per episode.
    
    Args:
        q_learning_results: Results from Q-learning
        sarsa_results: Results from SARSA
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Q-learning vs SARSA: Steps per Episode', fontsize=16)
    
    # Raw steps per episode
    q_steps = q_learning_results['stats']['episode_lengths']
    sarsa_steps = sarsa_results['stats']['episode_lengths']
    
    ax1.plot(q_steps, label='Q-learning', alpha=0.7)
    ax1.plot(sarsa_steps, label='SARSA', alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Steps')
    ax1.set_title('Steps per Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Smoothed steps per episode
    window_size = min(50, len(q_steps))
    
    smoothed_q = np.convolve(q_steps, np.ones(window_size)/window_size, mode='valid')
    smoothed_sarsa = np.convolve(sarsa_steps, np.ones(window_size)/window_size, mode='valid')
    
    ax2.plot(smoothed_q, label='Q-learning', linewidth=2)
    ax2.plot(smoothed_sarsa, label='SARSA', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Smoothed Steps')
    ax2.set_title(f'Smoothed Steps per Episode (window={window_size})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('q_vs_sarsa_steps_per_episode.png', dpi=300, bbox_inches='tight')


def visualize_policies_and_values(model: Model, q_learning_results: Dict, sarsa_results: Dict):
    """
    Visualize and compare final policies and value functions.
    
    Args:
        model: The environment model
        q_learning_results: Results from Q-learning
        sarsa_results: Results from SARSA
    """
    # Create figure with side-by-side policy and value function visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Q-learning vs SARSA: Final Policies and Value Functions', fontsize=16)
    
    # Convert policies from indices to Actions enum
    q_policy_enum = np.array([Actions(a) for a in q_learning_results['policy']])
    sarsa_policy_enum = np.array([Actions(a) for a in sarsa_results['policy']])
    
    # Prepare value functions
    def prepare_value_func(V):
        V_copy = V.copy()
        V_copy[np.isneginf(V_copy)] = -1000  # Replace -inf with large negative value
        return V_copy
    
    # Plot Q-learning policy and value function
    plot_vp(model, prepare_value_func(q_learning_results['V']), q_policy_enum, ax=axes[0])
    axes[0].set_title('Q-learning: Value Function & Policy')
    
    # Plot SARSA policy and value function
    plot_vp(model, prepare_value_func(sarsa_results['V']), sarsa_policy_enum, ax=axes[1])
    axes[1].set_title('SARSA: Value Function & Policy')
    
    # Find policy differences (we'll use this for the summary but not visualize it)
    policy_diff = q_learning_results['policy'] != sarsa_results['policy']
    diff_states = np.where(policy_diff)[0]
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('q_vs_sarsa_policies_values.png', dpi=300, bbox_inches='tight')
    
    # Return the diff_states for potential use in the summary
    return diff_states


def plot_q_value_heatmaps(model: Model, algorithm_name: str, Q_values: np.ndarray):
    """
    Create heatmap visualizations of Q-values for each action.
    
    Args:
        model: The environment model
        algorithm_name: Name of the algorithm (for title and filename)
        Q_values: Q-values to visualize
    """
    # Determine grid dimensions
    all_cells = set()
    for s in range(model.num_states):
        if s != model.fictional_end_state:
            cell = model.state2cell(s)
            all_cells.add(cell)
    
    if not all_cells:
        print("Warning: No valid cells found in the model.")
        return
    
    max_row = max(cell[0] for cell in all_cells)
    max_col = max(cell[1] for cell in all_cells)
    
    num_rows = max_row + 1
    num_cols = max_col + 1
    num_actions = len(Actions)
    
    # Create figure
    fig, axes = plt.subplots(1, num_actions, figsize=(5*num_actions, 4))
    fig.suptitle(f"{algorithm_name}: Q-value Heatmaps by Action", fontsize=16)
    
    # Define action names
    action_names = {
        Actions.UP: "Up",
        Actions.RIGHT: "Right",
        Actions.DOWN: "Down",
        Actions.LEFT: "Left"
    }
    
    # For each action, create a heatmap
    for a in range(num_actions):
        # Extract Q-values for this action and reshape to grid
        q_values = np.zeros((num_rows, num_cols))
        q_values.fill(np.nan)  # Fill with NaN for cells that aren't states
        
        # Fill in the Q-values for valid states
        for s in range(model.num_states):
            if s != model.fictional_end_state:
                row, col = model.state2cell(s)
                q_values[row, col] = Q_values[s, a]
        
        # Create mask for non-states
        mask = np.ones_like(q_values, dtype=bool)
        for s in range(model.num_states):
            if s != model.fictional_end_state:
                row, col = model.state2cell(s)
                mask[row, col] = False
        
        # Plot the heatmap
        ax = axes[a]
        cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
        sns.heatmap(q_values, 
                    mask=mask, 
                    ax=ax, 
                    cmap=cmap,
                    vmin=np.nanmin(q_values),
                    vmax=np.nanmax(q_values),
                    annot=True, 
                    fmt=".2f", 
                    cbar=True,
                    cbar_kws={'label': 'Q-value'})
        
        # Mark special cells
        start_row, start_col = model.state2cell(model.start_state)
        goal_row, goal_col = model.state2cell(model.goal_state)
        
        ax.add_patch(plt.Rectangle((start_col, start_row), 1, 1, fill=False, 
                                 edgecolor='blue', linewidth=2, label='Start'))
        ax.add_patch(plt.Rectangle((goal_col, goal_row), 1, 1, fill=False, 
                                 edgecolor='green', linewidth=2, label='Goal'))
        
        # Set title and labels
        ax.set_title(f"Action: {action_names[Actions(a)]}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{algorithm_name.lower().replace(" ", "_")}_q_heatmaps.png', dpi=300, bbox_inches='tight')


def generate_summary(model: Model, q_learning_results: Dict, sarsa_results: Dict):
    """
    Generate and save a summary of the comparison.
    
    Args:
        model: The environment model
        q_learning_results: Results from Q-learning
        sarsa_results: Results from SARSA
    """
    # Calculate key metrics
    q_train_time = q_learning_results['training_time']
    sarsa_train_time = sarsa_results['training_time']
    
    q_avg_return = np.mean(q_learning_results['stats']['episode_returns'][-100:])
    sarsa_avg_return = np.mean(sarsa_results['stats']['episode_returns'][-100:])
    
    q_avg_steps = np.mean(q_learning_results['stats']['episode_lengths'][-100:])
    sarsa_avg_steps = np.mean(sarsa_results['stats']['episode_lengths'][-100:])
    
    policy_diff = q_learning_results['policy'] != sarsa_results['policy']
    num_diff_states = np.sum(policy_diff)
    diff_pct = (num_diff_states / model.num_states) * 100
    
    # Create summary text
    summary = f"""
Q-learning vs SARSA Comparison Summary
=====================================

Environment: {model.__class__.__name__} - Small World
Grid size: {model.world.num_rows}x{model.world.num_cols}
Discount factor (γ): {model.gamma}

Hyperparameters:
- Learning rate (α): {q_learning_results['stats']['final_alpha']}
- Exploration rate (ε): {q_learning_results['stats']['final_epsilon']}
- Number of episodes: {len(q_learning_results['stats']['episode_returns'])}
- Maximum steps per episode: Used in training but not tracked in stats

Performance Metrics:
- Training time:
  * Q-learning: {q_train_time:.2f}s
  * SARSA: {sarsa_train_time:.2f}s
  * Faster algorithm: {'Q-learning' if q_train_time < sarsa_train_time else 'SARSA'} (by {abs(q_train_time - sarsa_train_time):.2f}s)

- Average return (last 100 episodes):
  * Q-learning: {q_avg_return:.2f}
  * SARSA: {sarsa_avg_return:.2f}
  * Better algorithm: {'Q-learning' if q_avg_return > sarsa_avg_return else 'SARSA'} (by {abs(q_avg_return - sarsa_avg_return):.2f})

- Average steps per episode (last 100 episodes):
  * Q-learning: {q_avg_steps:.2f}
  * SARSA: {sarsa_avg_steps:.2f}
  * More efficient algorithm: {'Q-learning' if q_avg_steps < sarsa_avg_steps else 'SARSA'} (by {abs(q_avg_steps - sarsa_avg_steps):.2f} steps)

Policy Differences:
- Number of states with different policies: {num_diff_states} ({diff_pct:.2f}%)

Analysis:
- Convergence: {'Q-learning' if q_train_time < sarsa_train_time else 'SARSA'} converged faster
- Final performance: {'Q-learning' if q_avg_return > sarsa_avg_return else 'SARSA'} achieved higher returns
- Policy similarity: {'High' if diff_pct < 10 else 'Moderate' if diff_pct < 30 else 'Low'} ({100-diff_pct:.2f}% of states have the same policy)

Theoretical Explanation:
Q-learning and SARSA differ in how they update their value estimates:
- Q-learning (off-policy) updates based on the maximum Q-value of the next state
- SARSA (on-policy) updates based on the actual next action chosen by the current policy

This difference leads to:
1. Q-learning tends to find more optimal policies faster but may be less stable
2. SARSA tends to be more conservative in risky environments since it accounts for exploration during learning
3. In environments with minimal risk or clear optimal paths, both algorithms often converge to similar policies
"""
    
    # Create results directory if it doesn't exist
    results_dir = "/Users/marinafranca/Desktop/gridworld-rl/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save summary to file in the results directory
    summary_file = os.path.join(results_dir, 'q_vs_sarsa_comparison_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    # Print summary to console
    print(summary)
    print(f"\nSummary saved to: {summary_file}")


def main():
    """Main function to run the comparison"""
    # Create model
    model = Model(small_world)
    
    # Best hyperparameters (example values - replace with actual best values from tuning)
    best_hyperparams = {
        'alpha': 0.5,
        'epsilon': 0.01,
        'gamma': 0.95,
        'num_episodes': 1000,
        'max_steps': 50
    }
    
    # Run the comparison
    q_learning_results, sarsa_results = run_comparison(model, best_hyperparams)
    
    print("\nComparison completed. All visualizations and summary have been saved.")


if __name__ == "__main__":
    main() 