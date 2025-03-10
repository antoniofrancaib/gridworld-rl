#!/usr/bin/env python3
"""
sarsa_comparison.py

Script for comparing SARSA vs. Expected SARSA with a fixed set of hyperparameters,
based on previously chosen or default values, without re-running the entire 81-combination search.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
from tqdm import tqdm
from typing import Optional, Dict, Tuple, List

# Adjust the path if needed so Python can find your environment code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.model import Model, Actions
from environment.world_config import small_world
from utils.plot_vp import plot_vp

class SarsaAgent:
    """
    Same SARSA agent as in hyperparameter_tuning.py, 
    but also supports Expected SARSA via a flag.
    """
    
    def __init__(
        self, 
        model: Model, 
        alpha: float = 0.5, 
        epsilon: float = 0.01,
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
    
    def expected_sarsa_update(self, s: int, a: int, r: float, s_next: int) -> float:
        current_q = self.Q[s, a]
        best_next_action = np.argmax(self.Q[s_next])
        
        expected_q = 0
        for a_next in range(len(Actions)):
            if a_next == best_next_action:
                prob = (1 - self.epsilon) + (self.epsilon / len(Actions))
            else:
                prob = self.epsilon / len(Actions)
            expected_q += prob * self.Q[s_next, a_next]
        
        td_target = r + self.gamma * expected_q
        td_error = td_target - current_q
        self.Q[s, a] += self.alpha * td_error
        return td_error
    
    def update_hyperparameters(self, episode: int):
        if self.epsilon_decay is not None:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon_init * (1 / (1 + self.epsilon_decay * episode))
            )
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
        expected_sarsa: bool = False,
        verbose: bool = True,
        early_stopping: Optional[Dict] = None
    ):
        algo_name = "Expected SARSA" if expected_sarsa else "SARSA"
        if verbose:
            print(f"Training {algo_name} for {num_episodes} episodes (max {max_steps} steps each)")
        
        start_time = time.time()
        
        for episode in tqdm(range(num_episodes), desc=f"Training {algo_name}", disable=not verbose):
            s = self.model.start_state
            a = self.epsilon_greedy_policy(s)
            
            episode_return = 0
            step = 0
            done = False
            
            while not done and step < max_steps:
                p_s_next = [
                    self.model.transition_probability(s, s_next_candidate, Actions(a))
                    for s_next_candidate in range(self.model.num_states)
                ]
                s_next = np.random.choice(self.model.num_states, p=p_s_next)
                r = self.model.reward(s, Actions(a))
                
                if s_next == self.model.fictional_end_state or s_next == self.model.goal_state:
                    done = True
                
                if not expected_sarsa:
                    # SARSA
                    a_next = self.epsilon_greedy_policy(s_next)
                    delta = self.sarsa_update(s, a, r, s_next, a_next)
                    s, a = s_next, a_next
                else:
                    # Expected SARSA
                    delta = self.expected_sarsa_update(s, a, r, s_next)
                    s = s_next
                    a = self.epsilon_greedy_policy(s)
                
                episode_return += r
                step += 1
                self.q_changes.append(abs(delta))
            
            self.episode_lengths.append(step)
            self.episode_returns.append(episode_return)
            self.update_hyperparameters(episode)
            
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
    
    def visualize_learning(self, title_suffix=""):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"SARSA Learning Progress {title_suffix}", fontsize=16)
        
        # Episode lengths
        axes[0, 0].plot(self.episode_lengths)
        axes[0, 0].set_title("Episode Lengths")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Length")
        
        # Episode returns
        axes[0, 1].plot(self.episode_returns)
        axes[0, 1].set_title("Episode Returns")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Return")
        
        # Smoothed returns
        window_size = min(50, len(self.episode_returns))
        smoothed = np.convolve(self.episode_returns, np.ones(window_size)/window_size, mode='valid')
        axes[1, 0].plot(smoothed)
        axes[1, 0].set_title(f"Smoothed Returns (window={window_size})")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Return")
        
        # TD Errors
        axes[1, 1].plot(self.q_changes)
        axes[1, 1].set_title("TD Errors")
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_xlabel("Update")
        axes[1, 1].set_ylabel("Absolute TD Error")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig
        
    def plot_q_value_heatmaps(self, title_prefix=""):
        """
        Create a visualization of Q-values for each action.
        
        Args:
            title_prefix: Optional prefix for the title (e.g., "SARSA" or "Expected SARSA")
        """
        # Get grid dimensions from the model
        num_states = self.model.num_states
        num_actions = len(Actions)
        
        # Create figure
        fig, axes = plt.subplots(1, num_actions, figsize=(5*num_actions, 4))
        
        # Set a single combined title
        if title_prefix:
            fig.suptitle(f"{title_prefix} Q-value Heatmaps by Action", fontsize=16)
        else:
            fig.suptitle("Q-value Heatmaps by Action", fontsize=16)
        
        # Action names for subplot titles
        action_names = {
            Actions.UP: "Up",
            Actions.RIGHT: "Right", 
            Actions.DOWN: "Down",
            Actions.LEFT: "Left"
        }
        
        # Extract grid dimensions from model
        # Find all valid cells
        all_cells = set()
        for s in range(num_states):
            if s != self.model.fictional_end_state:
                all_cells.add(self.model.state2cell(s))
        
        if not all_cells:
            print("Warning: No valid cells found")
            return fig
            
        max_row = max(cell[0] for cell in all_cells)
        max_col = max(cell[1] for cell in all_cells)
        num_rows = max_row + 1
        num_cols = max_col + 1
        
        # Plot heatmaps for each action
        for a in range(num_actions):
            # Create Q-value grid
            q_values = np.zeros((num_rows, num_cols))
            q_values.fill(np.nan)  # NaN for non-states
            
            # Create mask for non-states
            mask = np.ones_like(q_values, dtype=bool)
            
            # Fill in Q-values for valid states
            for s in range(num_states):
                if s != self.model.fictional_end_state:
                    row, col = self.model.state2cell(s)
                    q_values[row, col] = self.Q[s, a]
                    mask[row, col] = False
            
            # Plot the heatmap
            ax = axes[a]
            cmap = plt.cm.RdYlGn
            sns.heatmap(
                q_values,
                mask=mask,
                ax=ax,
                cmap=cmap,
                vmin=np.nanmin(q_values),
                vmax=np.nanmax(q_values),
                annot=True,
                fmt=".2f",
                cbar=True,
                cbar_kws={}
            )
            
            # Mark special cells
            start_row, start_col = self.model.state2cell(self.model.start_state)
            goal_row, goal_col = self.model.state2cell(self.model.goal_state)
            
            # Add rectangles for start and goal
            ax.add_patch(plt.Rectangle((start_col, start_row), 1, 1, fill=False, 
                                      edgecolor='blue', linewidth=3, label='Start'))
            ax.add_patch(plt.Rectangle((goal_col, goal_row), 1, 1, fill=False, 
                                      edgecolor='green', linewidth=3, label='Goal'))
            
            # Set subplot title and labels
            ax.set_title(f"Action: {action_names[Actions(a)]}")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig


def run_sarsa_comparison(
    model: Model, 
    num_episodes: int = 500, 
    max_steps: int = 50,
    alpha: float = 0.5,
    epsilon: float = 0.01,
    epsilon_decay: Optional[float] = 0.0,
    use_expected_sarsa: bool = True
):
    """
    Compare SARSA vs. Expected SARSA with a fixed set of hyperparameters.
    """
    print("===== Running SARSA vs. Expected SARSA Comparison =====")
    print(f"Environment: {model.__class__.__name__}")
    print(f"Hyperparams: alpha={alpha}, epsilon={epsilon}, episodes={num_episodes}, steps={max_steps}")
    
    # 1) SARSA
    sarsa_agent = SarsaAgent(model, alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay)
    sarsa_policy, sarsa_q, sarsa_stats = sarsa_agent.train(
        num_episodes=num_episodes, 
        max_steps=max_steps,
        expected_sarsa=False,
        verbose=True
    )
    sarsa_V = np.max(sarsa_q, axis=1)
    
    # 2) Expected SARSA
    results = {
        'SARSA': {
            'policy': sarsa_policy,
            'Q': sarsa_q,
            'V': sarsa_V,
            'stats': sarsa_stats
        }
    }
    
    # Plot SARSA Q-value heatmaps with proper title
    sarsa_agent.plot_q_value_heatmaps(title_prefix="SARSA")
    plt.savefig("sarsa_q_heatmaps.png")
    
    if use_expected_sarsa:
        exp_sarsa_agent = SarsaAgent(model, alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay)
        exp_policy, exp_q, exp_stats = exp_sarsa_agent.train(
            num_episodes=num_episodes, 
            max_steps=max_steps,
            expected_sarsa=True,
            verbose=True
        )
        exp_V = np.max(exp_q, axis=1)
        
        results['Expected SARSA'] = {
            'policy': exp_policy,
            'Q': exp_q,
            'V': exp_V,
            'stats': exp_stats
        }
        
        # Plot Expected SARSA Q-value heatmaps with proper title
        exp_sarsa_agent.plot_q_value_heatmaps(title_prefix="Expected SARSA")
        plt.savefig("expected_sarsa_q_heatmaps.png")
        
        # Plot learning curves (updated - more comprehensive)
        # --------------------------------------------------------------
        # 1. Learning Curves: Episode Returns vs. Episodes
        plt.figure(figsize=(12, 6))
        
        # Raw episode returns
        plt.plot(sarsa_stats['episode_returns'], alpha=0.3, color='blue', label='SARSA (raw)')
        plt.plot(exp_stats['episode_returns'], alpha=0.3, color='red', label='Expected SARSA (raw)')
        
        # Smoothed returns
        window_size = min(50, len(sarsa_stats['episode_returns']))
        sarsa_smooth = np.convolve(sarsa_stats['episode_returns'], np.ones(window_size)/window_size, mode='valid')
        exp_smooth = np.convolve(exp_stats['episode_returns'], np.ones(window_size)/window_size, mode='valid')
        
        plt.plot(range(window_size-1, len(sarsa_stats['episode_returns'])), 
                 sarsa_smooth, 
                 linewidth=2, 
                 color='blue', 
                 label=f'SARSA (smoothed, window={window_size})')
        plt.plot(range(window_size-1, len(exp_stats['episode_returns'])), 
                 exp_smooth, 
                 linewidth=2, 
                 color='red', 
                 label=f'Expected SARSA (smoothed, window={window_size})')
        
        plt.title("Learning Curves: Episode Returns")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("sarsa_vs_expected_learning_curves.png")
        
        # 2. Steps per Episode Comparison
        # --------------------------------------------------------------
        plt.figure(figsize=(12, 8))
        
        # Two subplots: one for each algorithm
        plt.subplot(2, 1, 1)
        plt.plot(sarsa_stats['episode_lengths'], alpha=0.5, label='Raw')
        
        # Add smoothed line for trend visualization
        window_size = min(50, len(sarsa_stats['episode_lengths']))
        if window_size > 0:
            smoothed_steps = np.convolve(
                sarsa_stats['episode_lengths'], 
                np.ones(window_size) / window_size, 
                mode='valid'
            )
            plt.plot(range(window_size-1, len(sarsa_stats['episode_lengths'])), 
                     smoothed_steps, 
                     'r-', 
                     linewidth=2, 
                     label=f'Moving Average (window={window_size})')
            
        # Add horizontal line at mean of last 50 episodes
        if len(sarsa_stats['episode_lengths']) >= 50:
            mean_last_50 = np.mean(sarsa_stats['episode_lengths'][-50:])
            plt.axhline(y=mean_last_50, color='g', linestyle='--', 
                        label=f'Mean of last 50 episodes: {mean_last_50:.1f}')
        
        plt.title("SARSA: Steps per Episode")
        plt.ylabel("Steps")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Expected SARSA subplot
        plt.subplot(2, 1, 2)
        plt.plot(exp_stats['episode_lengths'], alpha=0.5, label='Raw')
        
        # Add smoothed line for trend visualization
        window_size = min(50, len(exp_stats['episode_lengths']))
        if window_size > 0:
            smoothed_steps = np.convolve(
                exp_stats['episode_lengths'], 
                np.ones(window_size) / window_size, 
                mode='valid'
            )
            plt.plot(range(window_size-1, len(exp_stats['episode_lengths'])), 
                     smoothed_steps, 
                     'r-', 
                     linewidth=2, 
                     label=f'Moving Average (window={window_size})')
            
        # Add horizontal line at mean of last 50 episodes
        if len(exp_stats['episode_lengths']) >= 50:
            mean_last_50 = np.mean(exp_stats['episode_lengths'][-50:])
            plt.axhline(y=mean_last_50, color='g', linestyle='--', 
                        label=f'Mean of last 50 episodes: {mean_last_50:.1f}')
        
        plt.title("Expected SARSA: Steps per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("sarsa_vs_expected_steps_per_episode.png")
    
    # Save results
    results_dir = "/Users/marinafranca/Desktop/gridworld-rl/results"
    os.makedirs(results_dir, exist_ok=True)
    txt_file = os.path.join(results_dir, "sarsa_comparison_results.txt")
    with open(txt_file, 'a') as f:
        f.write("\n\n===== SARSA vs. Expected SARSA (Fixed Hyperparams) =====\n")
        f.write(f"alpha={alpha}, epsilon={epsilon}, episodes={num_episodes}, steps={max_steps}\n")
        f.write(f"SARSA training time: {sarsa_stats['training_time']:.2f} s\n")
        if use_expected_sarsa:
            f.write(f"Expected SARSA training time: {exp_stats['training_time']:.2f} s\n")
    
    print("Comparison plots saved. Detailed stats appended to sarsa_comparison_results.txt")
    return results


def create_alpha_epsilon_heatmaps(
    model: Model,
    num_episodes: int = 500,
    max_steps: int = 100,
    alphas: List[float] = [0.01, 0.1, 0.5],
    epsilons: List[float] = [0.01, 0.1, 0.3],
    epsilon_decay: float = 0.0
):
    """
    Create alpha-epsilon heatmaps for both SARSA and Expected SARSA, showing the final
    average return for each combination of hyperparameters.
    
    Args:
        model: The environment model to use
        num_episodes: Number of episodes to train each agent
        max_steps: Maximum steps per episode
        alphas: List of alpha values to test
        epsilons: List of epsilon values to test
        epsilon_decay: Epsilon decay rate
        
    Returns:
        Tuple of results for SARSA and Expected SARSA
    """
    print("===== Creating Alpha-Epsilon Heatmaps =====")
    print(f"Testing {len(alphas)}×{len(epsilons)} combinations for both algorithms")
    
    # Initialize arrays to store results
    sarsa_results = np.zeros((len(alphas), len(epsilons)))
    exp_sarsa_results = np.zeros((len(alphas), len(epsilons)))
    
    # Run both algorithms for each alpha-epsilon combination
    for i, alpha in enumerate(alphas):
        for j, epsilon in enumerate(epsilons):
            print(f"\nTraining with α={alpha}, ε={epsilon}")
            
            # SARSA
            sarsa_agent = SarsaAgent(
                model=model,
                alpha=alpha,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay
            )
            _, _, sarsa_stats = sarsa_agent.train(
                num_episodes=num_episodes,
                max_steps=max_steps,
                expected_sarsa=False,
                verbose=False
            )
            
            # Expected SARSA
            exp_sarsa_agent = SarsaAgent(
                model=model,
                alpha=alpha,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay
            )
            _, _, exp_stats = exp_sarsa_agent.train(
                num_episodes=num_episodes,
                max_steps=max_steps,
                expected_sarsa=True,
                verbose=False
            )
            
            # Calculate final average returns (over last 50 episodes or 10% of episodes)
            window_size = min(50, max(10, int(num_episodes * 0.1)))
            sarsa_avg_return = np.mean(sarsa_stats['episode_returns'][-window_size:])
            exp_avg_return = np.mean(exp_stats['episode_returns'][-window_size:])
            
            # Store results
            sarsa_results[i, j] = sarsa_avg_return
            exp_sarsa_results[i, j] = exp_avg_return
    
    # Create heatmap visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("SARSA vs Expected SARSA: Final Average Returns", fontsize=16)
    
    # Find global min and max for consistent color scaling
    vmin = min(np.min(sarsa_results), np.min(exp_sarsa_results))
    vmax = max(np.max(sarsa_results), np.max(exp_sarsa_results))
    
    # SARSA heatmap
    sns.heatmap(
        sarsa_results,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        ax=axes[0],
        xticklabels=epsilons,
        yticklabels=alphas
    )
    axes[0].set_title("SARSA")
    axes[0].set_xlabel("Epsilon (ε)")
    axes[0].set_ylabel("Alpha (α)")
    
    # Expected SARSA heatmap
    sns.heatmap(
        exp_sarsa_results,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        ax=axes[1],
        xticklabels=epsilons,
        yticklabels=alphas
    )
    axes[1].set_title("Expected SARSA")
    axes[1].set_xlabel("Epsilon (ε)")
    axes[1].set_ylabel("Alpha (α)")
    
    plt.tight_layout()
    plt.savefig("sarsa_vs_expected_heatmaps.png", dpi=300)
    
    print("Alpha-Epsilon heatmaps created and saved")
    return sarsa_results, exp_sarsa_results


if __name__ == "__main__":
    # Create model with small_world configuration
    model = Model(small_world)
    
    # 1. Run the standard comparison with fixed parameters
    print("\n===== Running Standard Comparison =====")
    results = run_sarsa_comparison(
        model=model,
        num_episodes=500,
        max_steps=200,
        alpha=0.1,
        epsilon=0.1,
        epsilon_decay=0.0,
        use_expected_sarsa=True
    )
    
    # 2. Create alpha-epsilon heatmaps
    print("\n===== Creating Alpha-Epsilon Heatmaps =====")
    sarsa_results, exp_results = create_alpha_epsilon_heatmaps(
        model=model,
        num_episodes=300,  # Using fewer episodes for faster computation
        max_steps=100,
        alphas=[0.01, 0.1, 0.5],
        epsilons=[0.01, 0.1, 0.3]
    )
    
    print("\nAll analyses complete!")
    plt.show()
