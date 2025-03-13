#!/usr/bin/env python3
"""
q_learning_hyperparameter_tuning.py

Script for systematically exploring multiple hyperparameter combinations for Q-learning
and visualizing the final average returns in heatmaps for different gamma values.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
from tqdm import tqdm
from typing import Optional, Dict

# Adjust the path if needed so Python can find your environment code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.model import Model, Actions
from environment.world_config import small_world
from utils.plot_vp import plot_vp
from algorithms.q_learning import QLearningAgent

def hyperparameter_tuning_q_learning(model: Model, gamma: float):
    """
    Demonstrate the effect of different hyperparameters on Q-learning using heatmaps.
    
    Creates a 3×3 grid of heatmaps, each showing final average returns for different
    combinations of alpha and epsilon, for a specific (num_episodes, max_steps) pair.
    
    Args:
        model: The environment model
        gamma: The discount factor to use
    
    Returns:
        best_performance: Dictionary with the best hyperparameter combination
    """
    # Set the model's gamma value
    original_gamma = model.gamma
    model.gamma = gamma
    
    alphas = [0.01, 0.1, 0.5]
    epsilons = [0.01, 0.1, 0.3]
    num_episodes_values = [100, 500, 1000]
    max_steps_values = [50, 200, 500]
    
    # 4D array: [e_idx, s_idx, a_idx, eps_idx]
    results = np.zeros((len(num_episodes_values), len(max_steps_values), len(alphas), len(epsilons)))
    
    best_performance = {
        'gamma': gamma,
        'alpha': None,
        'epsilon': None,
        'num_episodes': None,
        'max_steps': None,
        'return': -float('inf')
    }
    
    for e_idx, num_episodes in enumerate(num_episodes_values):
        for s_idx, max_steps in enumerate(max_steps_values):
            for a_idx, alpha in enumerate(alphas):
                for eps_idx, epsilon in enumerate(epsilons):
                    print(f"\nTesting Q-learning with γ={gamma}, α={alpha}, ε={epsilon}, "
                          f"episodes={num_episodes}, max_steps={max_steps}")
                    
                    agent = QLearningAgent(model=model, alpha=alpha, epsilon=epsilon)
                    
                    _, _, stats = agent.train(
                        num_episodes=num_episodes,
                        max_steps=max_steps,
                        verbose=False
                    )
                    
                    final_window = min(50, max(10, int(num_episodes * 0.2)))
                    final_return_avg = np.mean(stats['episode_returns'][-final_window:])
                    
                    results[e_idx, s_idx, a_idx, eps_idx] = final_return_avg
                    
                    if final_return_avg > best_performance['return']:
                        best_performance['alpha'] = alpha
                        best_performance['epsilon'] = epsilon
                        best_performance['num_episodes'] = num_episodes
                        best_performance['max_steps'] = max_steps
                        best_performance['return'] = final_return_avg
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(f"Q-learning Hyperparameter Tuning: Final Average Returns (γ={gamma})", fontsize=20)
    
    vmin = np.min(results)
    vmax = np.max(results)
    
    for e_idx, num_episodes in enumerate(num_episodes_values):
        for s_idx, max_steps in enumerate(max_steps_values):
            ax = axes[e_idx, s_idx]
            heatmap_data = results[e_idx, s_idx]
            
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                xticklabels=epsilons,
                yticklabels=alphas,
                cbar=False
            )
            
            ax.set_title(f"Episodes={num_episodes}, Steps={max_steps}", fontsize=12)
            ax.set_xlabel("Epsilon (ε)" if e_idx == 2 else "")
            ax.set_ylabel("Alpha (α)" if s_idx == 0 else "")
            
            if (best_performance['num_episodes'] == num_episodes and 
                best_performance['max_steps'] == max_steps):
                best_a_idx = alphas.index(best_performance['alpha'])
                best_eps_idx = epsilons.index(best_performance['epsilon'])
                ax.add_patch(plt.Rectangle(
                    (best_eps_idx, best_a_idx), 1, 1, 
                    fill=False, edgecolor='red', linewidth=3, clip_on=False
                ))
    
    # Single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Final Average Return', rotation=270, labelpad=20)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save the figure with gamma in the filename
    plt.savefig(f"q_learning_hyperparameter_heatmaps_gamma_{gamma}.png", dpi=300, bbox_inches='tight')
    
    # Save best combo to file
    results_dir = "/Users/marinafranca/Desktop/gridworld-rl/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"q_learning_gamma_{gamma}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"===== Q-learning Hyperparameter Tuning Results (γ={gamma}) =====\n")
        f.write("Best parameter combination:\n")
        f.write(f"  γ={gamma}\n")
        f.write(f"  α={best_performance['alpha']}\n")
        f.write(f"  ε={best_performance['epsilon']}\n")
        f.write(f"  Episodes={best_performance['num_episodes']}\n")
        f.write(f"  Max steps={best_performance['max_steps']}\n")
        f.write(f"  Return={best_performance['return']:.2f}\n")
    
    print(f"\n===== Q-learning Hyperparameter Tuning Results (γ={gamma}) =====")
    print("Best parameter combination:")
    print(f"  γ={gamma}")
    print(f"  α={best_performance['alpha']}")
    print(f"  ε={best_performance['epsilon']}")
    print(f"  Episodes={best_performance['num_episodes']}")
    print(f"  Max steps={best_performance['max_steps']}")
    print(f"  Return={best_performance['return']:.2f}")
    
    # Restore original gamma
    model.gamma = original_gamma
    
    return best_performance


def compare_gamma_values(model: Model, gamma_values=None):
    """
    Run hyperparameter tuning for different gamma values and compare the results.
    
    Args:
        model: The environment model
        gamma_values: List of gamma values to test (default: [0.9, 0.95, 1.0])
    
    Returns:
        gamma_results: Dictionary with best performance for each gamma value
    """
    if gamma_values is None:
        gamma_values = [0.9, 0.95, 1.0]
    
    gamma_results = {}
    
    for gamma in gamma_values:
        print(f"\n{'='*80}")
        print(f"Running Q-learning hyperparameter tuning with γ={gamma}")
        print(f"{'='*80}")
        
        best_params = hyperparameter_tuning_q_learning(model, gamma)
        gamma_results[gamma] = best_params
    
    # Create summary plot comparing gamma values
    plt.figure(figsize=(10, 6))
    
    gammas = list(gamma_results.keys())
    returns = [gamma_results[g]['return'] for g in gammas]
    
    plt.bar(gammas, returns, width=0.03)
    plt.xlabel('Gamma (γ)')
    plt.ylabel('Best Average Return')
    plt.title('Effect of Discount Factor (γ) on Q-learning Performance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text annotations for best hyperparameters
    for i, gamma in enumerate(gammas):
        best_alpha = gamma_results[gamma]['alpha']
        best_epsilon = gamma_results[gamma]['epsilon']
        best_return = gamma_results[gamma]['return']
        
        plt.text(gamma, best_return + max(returns)*0.02, 
                 f"α={best_alpha}, ε={best_epsilon}",
                 ha='center', va='bottom', fontsize=9)
    
    plt.savefig("q_learning_gamma_comparison.png", dpi=300, bbox_inches='tight')
    
    # Print summary of gamma comparison
    print("\n" + "="*80)
    print("SUMMARY: Effect of Discount Factor (γ) on Q-learning")
    print("="*80)
    
    # Determine optimal gamma
    best_gamma = max(gamma_results.keys(), key=lambda g: gamma_results[g]['return'])
    
    print(f"Optimal discount factor (γ): {best_gamma}")
    print(f"Corresponding hyperparameters:")
    print(f"  α={gamma_results[best_gamma]['alpha']}")
    print(f"  ε={gamma_results[best_gamma]['epsilon']}")
    print(f"  Return={gamma_results[best_gamma]['return']:.2f}")
    
    print("\nComparison of gamma values:")
    for gamma in sorted(gamma_results.keys()):
        print(f"  γ={gamma}: Return={gamma_results[gamma]['return']:.2f}, "
              f"α={gamma_results[gamma]['alpha']}, ε={gamma_results[gamma]['epsilon']}")
    
    return gamma_results


if __name__ == "__main__":
    from environment.model import Model
    from environment.world_config import small_world
    
    model = Model(small_world)
    print("Running Q-learning hyperparameter tuning for different gamma values...")
    gamma_results = compare_gamma_values(model, gamma_values=[0.9, 0.95, 1.0])
    print("\nDone. See the generated heatmaps and comparison plots.") 