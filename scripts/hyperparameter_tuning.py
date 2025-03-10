#!/usr/bin/env python3
"""
hyperparameter_tuning.py

Script for systematically exploring multiple hyperparameter combinations for SARSA
and visualizing the final average returns in heatmaps. This does NOT run Expected SARSA.
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

class SarsaAgent:
    """
    Implementation of SARSA (State-Action-Reward-State-Action) algorithm.
    
    Key parameters:
    - alpha: Learning rate
    - epsilon: Exploration rate
    - gamma: Discount factor (taken from model)
    """
    
    def __init__(
        self, 
        model: Model, 
        alpha: float = 0.1, 
        epsilon: float = 0.1,
        epsilon_decay: Optional[float] = None,
        epsilon_min: float = 0.01,
        alpha_decay: Optional[float] = None,
        alpha_min: float = 0.01
    ):
        self.model = model
        self.alpha = alpha
        self.alpha_init = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        
        self.epsilon = epsilon
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.gamma = model.gamma
        
        self.Q = np.zeros((model.num_states, len(Actions)))
        self.policy = np.zeros(model.num_states, dtype=int)
        
        # Statistics
        self.episode_lengths = []
        self.episode_returns = []
        self.q_changes = []
    
    def epsilon_greedy_policy(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.choice(len(Actions))
        else:
            return np.argmax(self.Q[state])
    
    def sarsa_update(self, s: int, a: int, r: float, s_next: int, a_next: int) -> float:
        current_q = self.Q[s, a]
        next_q = self.Q[s_next, a_next]
        td_target = r + self.gamma * next_q
        td_error = td_target - current_q
        self.Q[s, a] += self.alpha * td_error
        return td_error
    
    def update_hyperparameters(self, episode: int):
        # Decay epsilon
        if self.epsilon_decay is not None:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon_init * (1 / (1 + self.epsilon_decay * episode))
            )
        
        # Decay alpha
        if self.alpha_decay is not None:
            self.alpha = max(
                self.alpha_min,
                self.alpha_init * (1 / (1 + self.alpha_decay * episode))
            )
    
    def extract_policy(self):
        for s in range(self.model.num_states):
            self.policy[s] = np.argmax(self.Q[s])
        return self.policy
    
    def train(
        self, 
        num_episodes: int = 1000, 
        max_steps: int = 1000, 
        verbose: bool = True,
        early_stopping: Optional[Dict] = None
    ):
        start_time = time.time()
        
        for episode in tqdm(range(num_episodes), desc="Training SARSA"):
            s = self.model.start_state
            a = self.epsilon_greedy_policy(s)
            
            episode_return = 0
            step = 0
            done = False
            
            while not done and step < max_steps:
                # Sample next state
                p_s_next = [
                    self.model.transition_probability(s, s_next_candidate, Actions(a))
                    for s_next_candidate in range(self.model.num_states)
                ]
                s_next = np.random.choice(self.model.num_states, p=p_s_next)
                
                # Reward
                r = self.model.reward(s, Actions(a))
                
                if s_next == self.model.fictional_end_state or s_next == self.model.goal_state:
                    done = True
                
                # Next action
                a_next = self.epsilon_greedy_policy(s_next)
                
                # SARSA update
                delta = self.sarsa_update(s, a, r, s_next, a_next)
                
                s, a = s_next, a_next
                episode_return += r
                step += 1
                self.q_changes.append(abs(delta))
            
            self.episode_lengths.append(step)
            self.episode_returns.append(episode_return)
            self.update_hyperparameters(episode)
            
            # Optional early stopping
            if early_stopping and episode >= early_stopping.get('min_episodes', 0):
                window_size = early_stopping.get('window_size', 50)
                threshold = early_stopping.get('threshold', 0.01)
                
                if episode >= window_size:
                    recent_returns = self.episode_returns[-window_size:]
                    avg_return = np.mean(recent_returns)
                    std_return = np.std(recent_returns)
                    if std_return < threshold:
                        break
        
        training_time = time.time() - start_time
        policy = self.extract_policy()
        stats = {
            'episode_lengths': self.episode_lengths,
            'episode_returns': self.episode_returns,
            'q_changes': self.q_changes,
            'training_time': training_time
        }
        return policy, self.Q, stats

def hyperparameter_tuning_demo(model: Model):
    """
    Demonstrate the effect of different hyperparameters on SARSA learning using heatmaps.
    
    Creates a 3×3 grid of heatmaps, each showing final average returns for different
    combinations of alpha and epsilon, for a specific (num_episodes, max_steps) pair.
    """
    alphas = [0.01, 0.1, 0.5]
    epsilons = [0.01, 0.1, 0.3]
    num_episodes_values = [100, 500, 1000]
    max_steps_values = [50, 200, 500]
    
    # 4D array: [e_idx, s_idx, a_idx, eps_idx]
    results = np.zeros((len(num_episodes_values), len(max_steps_values), len(alphas), len(epsilons)))
    
    best_performance = {
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
                    print(f"\nTesting SARSA with α={alpha}, ε={epsilon}, "
                          f"episodes={num_episodes}, max_steps={max_steps}")
                    
                    agent = SarsaAgent(model=model, alpha=alpha, epsilon=epsilon)
                    
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
    fig.suptitle("SARSA Hyperparameter Tuning: Final Average Returns", fontsize=20)
    
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
    
    # Save the figure
    plt.savefig("sarsa_hyperparameter_heatmaps.png", dpi=300, bbox_inches='tight')
    
    # Save best combo to file
    results_dir = "/Users/marinafranca/Desktop/gridworld-rl/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "sarsa_comparison_results.txt")
    with open(results_file, 'a') as f:
        f.write("\n\n===== SARSA Hyperparameter Tuning Results =====\n")
        f.write("Best parameter combination:\n")
        f.write(f"  α={best_performance['alpha']}\n")
        f.write(f"  ε={best_performance['epsilon']}\n")
        f.write(f"  Episodes={best_performance['num_episodes']}\n")
        f.write(f"  Max steps={best_performance['max_steps']}\n")
        f.write(f"  Return={best_performance['return']:.2f}\n")
    
    print("\n===== Hyperparameter Tuning Results =====")
    print("Best parameter combination:")
    print(f"  α={best_performance['alpha']}")
    print(f"  ε={best_performance['epsilon']}")
    print(f"  Episodes={best_performance['num_episodes']}")
    print(f"  Max steps={best_performance['max_steps']}")
    print(f"  Return={best_performance['return']:.2f}")
    
    return best_performance


if __name__ == "__main__":
    from environment.model import Model
    from environment.world_config import small_world
    
    model = Model(small_world)
    print("Running hyperparameter tuning on small_world...")
    best_params = hyperparameter_tuning_demo(model)
    print("\nDone. See sarsa_hyperparameter_heatmaps.png for heatmaps.")
