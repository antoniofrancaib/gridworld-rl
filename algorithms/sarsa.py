import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import os
import sys
import seaborn as sns
from matplotlib.gridspec import GridSpecFromSubplotSpec

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.model import Model, Actions
from utils.plot_vp import plot_vp
from typing import List, Tuple, Dict, Optional, Callable


class SarsaAgent:
    """
    Implementation of SARSA (State-Action-Reward-State-Action) algorithm.
    
    SARSA is an on-policy TD control algorithm that learns the action-value function Q(s,a)
    and updates it based on the actual action taken in the next state.
    
    Key parameters:
    - alpha: Learning rate (determines how much new information overwrites old)
    - epsilon: Exploration rate (probability of taking a random action)
    - gamma: Discount factor (from the model, determines importance of future rewards)
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
        """
        Initialize the SARSA agent.
        
        Args:
            model: The environment model
            alpha: Learning rate (default: 0.1)
                - If too low: Learning is slow, requires many episodes to converge
                - If too high: Can oscillate or diverge, failing to converge to optimal Q-values
            epsilon: Exploration rate (default: 0.1)
                - If too low: May get stuck in suboptimal policies (exploitation dominates)
                - If too high: Too much random exploration, slow convergence
            epsilon_decay: Factor to decay epsilon after each episode (default: None)
            epsilon_min: Minimum epsilon value after decay (default: 0.01)
            alpha_decay: Factor to decay alpha after each episode (default: None)
            alpha_min: Minimum alpha value after decay (default: 0.01)
        """
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
        
        # Initialize Q-values: Q(s,a) for all state-action pairs
        # We create a 2D array: states x actions
        self.Q = np.zeros((model.num_states, len(Actions)))
        
        # Initialize policy derived from Q-values
        self.policy = np.zeros(model.num_states, dtype=int)
        
        # Statistics
        self.episode_lengths = []
        self.episode_returns = []
        self.q_changes = []
    
    def epsilon_greedy_policy(self, state: int) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(len(Actions))
        else:
            # Exploit: best action according to Q-values
            return np.argmax(self.Q[state])
    
    def sarsa_update(self, s: int, a: int, r: float, s_next: int, a_next: int) -> float:
        """
        Perform a SARSA update.
        
        Args:
            s: Current state
            a: Current action
            r: Reward received
            s_next: Next state
            a_next: Next action
            
        Returns:
            delta: TD error (change in Q-value)
        """
        # SARSA update rule: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        current_q = self.Q[s, a]
        next_q = self.Q[s_next, a_next]
        
        # TD target: r + γQ(s',a')
        td_target = r + self.gamma * next_q
        
        # TD error: r + γQ(s',a') - Q(s,a)
        td_error = td_target - current_q
        
        # Update Q-value
        self.Q[s, a] += self.alpha * td_error
        
        return td_error
    
    def expected_sarsa_update(self, s: int, a: int, r: float, s_next: int) -> float:
        """
        Perform an Expected SARSA update.
        
        Expected SARSA differs from regular SARSA by using the expected value of Q(s',a')
        instead of the actual next action's Q-value.
        
        Args:
            s: Current state
            a: Current action
            r: Reward received
            s_next: Next state
            
        Returns:
            delta: TD error (change in Q-value)
        """
        # Expected SARSA update rule: Q(s,a) ← Q(s,a) + α[r + γ∑_a' π(a'|s')Q(s',a') - Q(s,a)]
        current_q = self.Q[s, a]
        
        # Calculate expected value of next state
        # For epsilon-greedy policy:
        # With probability (1-epsilon), we take the best action
        # With probability epsilon, we take a random action (1/|A| for each action)
        best_next_action = np.argmax(self.Q[s_next])
        
        # Expected value calculation
        expected_q = 0
        for a_next in range(len(Actions)):
            # Probability of taking action a_next
            if a_next == best_next_action:
                # Best action is taken with probability (1-epsilon) + epsilon/|A|
                prob = (1 - self.epsilon) + (self.epsilon / len(Actions))
            else:
                # Non-best actions are taken with probability epsilon/|A|
                prob = self.epsilon / len(Actions)
            
            expected_q += prob * self.Q[s_next, a_next]
        
        # TD target: r + γ∑_a' π(a'|s')Q(s',a')
        td_target = r + self.gamma * expected_q
        
        # TD error: r + γ∑_a' π(a'|s')Q(s',a') - Q(s,a)
        td_error = td_target - current_q
        
        # Update Q-value
        self.Q[s, a] += self.alpha * td_error
        
        return td_error
    
    def update_hyperparameters(self, episode: int):
        """
        Update hyperparameters (epsilon, alpha) based on decay schedules.
        
        Args:
            episode: Current episode number
        """
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
        """
        Extract deterministic policy from learned Q-values.
        
        Returns:
            policy: Array of actions for each state
        """
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
        """
        Train the agent using SARSA or Expected SARSA.
        
        Args:
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            expected_sarsa: Whether to use Expected SARSA (True) or regular SARSA (False)
            verbose: Whether to print progress
            early_stopping: Optional dictionary with early stopping criteria:
                - 'min_episodes': Minimum number of episodes before checking for early stopping
                - 'window_size': Number of episodes to average returns over
                - 'threshold': Threshold for convergence
                
        Returns:
            policy: The learned policy
            Q: The learned Q-values
            stats: Dictionary with training statistics
        """
        algorithm_name = "Expected SARSA" if expected_sarsa else "SARSA"
        if verbose:
            print(f"Training {algorithm_name} for {num_episodes} episodes (max {max_steps} steps per episode)")
            print(f"  alpha={self.alpha_init} {'with decay' if self.alpha_decay else 'fixed'}")
            print(f"  epsilon={self.epsilon_init} {'with decay' if self.epsilon_decay else 'fixed'}")
            print(f"  gamma={self.gamma}")
        
        start_time = time.time()
        
        for episode in tqdm(range(num_episodes), desc=f"Training {algorithm_name}"):
            # Reset agent state
            s = self.model.start_state  # Start at initial state
            a = self.epsilon_greedy_policy(s)  # Select first action using policy
            
            # Track episode statistics
            episode_return = 0
            step = 0
            
            # Run episode
            done = False
            while not done and step < max_steps:
                # Take action, get reward and next state
                s_next = None
                
                # Sample next state based on transition probabilities
                p_s_next = [
                    self.model.transition_probability(s, s_next_candidate, Actions(a))
                    for s_next_candidate in range(self.model.num_states)
                ]
                
                # Sample next state
                s_next = np.random.choice(self.model.num_states, p=p_s_next)
                
                # Get reward
                r = self.model.reward(s, Actions(a))
                
                # Check if terminal state
                if s_next == self.model.fictional_end_state or s_next == self.model.goal_state:
                    done = True
                
                # Select next action using policy (only for regular SARSA)
                if not expected_sarsa:
                    a_next = self.epsilon_greedy_policy(s_next)
                    
                    # Update Q-values using SARSA
                    delta = self.sarsa_update(s, a, r, s_next, a_next)
                    
                    # Move to next state-action pair
                    s, a = s_next, a_next
                else:
                    # Update Q-values using Expected SARSA
                    delta = self.expected_sarsa_update(s, a, r, s_next)
                    
                    # Move to next state
                    s = s_next
                    # Select next action for execution (not for expectation calculation)
                    a = self.epsilon_greedy_policy(s)
                
                # Track statistics
                episode_return += r
                step += 1
                self.q_changes.append(abs(delta))
            
            # Record episode statistics
            self.episode_lengths.append(step)
            self.episode_returns.append(episode_return)
            
            # Update hyperparameters
            self.update_hyperparameters(episode)
            
            # Check for early stopping
            if early_stopping and episode >= early_stopping.get('min_episodes', 0):
                window_size = early_stopping.get('window_size', 50)
                threshold = early_stopping.get('threshold', 0.01)
                
                if episode >= window_size:
                    recent_returns = self.episode_returns[-window_size:]
                    avg_return = np.mean(recent_returns)
                    std_return = np.std(recent_returns)
                    
                    if std_return < threshold:
                        if verbose:
                            print(f"Early stopping at episode {episode}: return stabilized at {avg_return:.2f}±{std_return:.2f}")
                        break
        
        training_time = time.time() - start_time
        
        # Extract final policy
        policy = self.extract_policy()
        
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Final epsilon: {self.epsilon:.4f}")
            print(f"Final alpha: {self.alpha:.4f}")
            avg_return = np.mean(self.episode_returns[-100:])
            print(f"Average return over last 100 episodes: {avg_return:.2f}")
        
        # Return training statistics
        stats = {
            'episode_lengths': self.episode_lengths,
            'episode_returns': self.episode_returns,
            'q_changes': self.q_changes,
            'training_time': training_time,
            'final_epsilon': self.epsilon,
            'final_alpha': self.alpha
        }
        
        return policy, self.Q, stats
    
    def visualize_learning(self):
        """
        Visualize the learning progress.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("SARSA Learning Progress", fontsize=16)
        
        # Plot episode lengths
        axes[0, 0].plot(self.episode_lengths)
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Length")
        axes[0, 0].set_title("Episode Lengths")
        
        # Plot episode returns
        axes[0, 1].plot(self.episode_returns)
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Return")
        axes[0, 1].set_title("Episode Returns")
        
        # Smooth episode returns for better visualization
        window_size = min(50, len(self.episode_returns))
        smoothed_returns = np.convolve(
            self.episode_returns, 
            np.ones(window_size) / window_size, 
            mode='valid'
        )
        axes[1, 0].plot(smoothed_returns)
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Smoothed Return")
        axes[1, 0].set_title(f"Smoothed Returns (window={window_size})")
        
        # Plot Q-value changes
        axes[1, 1].plot(self.q_changes)
        axes[1, 1].set_xlabel("Update")
        axes[1, 1].set_ylabel("Absolute TD Error")
        axes[1, 1].set_title("TD Errors")
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def plot_steps_per_episode(self):
        """
        Create a dedicated plot for steps per episode to track agent efficiency.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_lengths)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Steps per Episode')
        
        # Add smoothed line for trend visualization
        window_size = min(50, len(self.episode_lengths))
        if window_size > 0:
            smoothed_steps = np.convolve(
                self.episode_lengths, 
                np.ones(window_size) / window_size, 
                mode='valid'
            )
            plt.plot(range(window_size-1, len(self.episode_lengths)), 
                     smoothed_steps, 
                     'r-', 
                     linewidth=2, 
                     label=f'Moving Average (window={window_size})')
            
        # Add horizontal line at mean of last 50 episodes
        if len(self.episode_lengths) >= 50:
            mean_last_50 = np.mean(self.episode_lengths[-50:])
            plt.axhline(y=mean_last_50, color='g', linestyle='--', 
                        label=f'Mean of last 50 episodes: {mean_last_50:.1f}')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_q_value_heatmaps(self):
        """
        Create heatmap visualizations of Q-values for each action.
        
        This shows how the agent values each action across different states in the grid.
        """
        # Get grid dimensions from the model
        # The model doesn't expose world_config directly, so we need to infer dimensions
        
        # First find all valid cells to determine grid dimensions
        all_cells = set()
        for s in range(self.model.num_states):
            if s != self.model.fictional_end_state:  # Skip fictional end state
                # Get the cell coordinates for this state
                cell = self.model.state2cell(s)
                all_cells.add(cell)
        
        # Determine grid dimensions from the valid cells
        if not all_cells:
            print("Warning: No valid cells found in the model.")
            return None
            
        # Find max row and column
        max_row = max(cell[0] for cell in all_cells)
        max_col = max(cell[1] for cell in all_cells)
        
        num_rows = max_row + 1  # +1 because 0-indexed
        num_cols = max_col + 1  # +1 because 0-indexed
        num_actions = len(Actions)
        
        # Create a figure with subplots for each action
        fig, axes = plt.subplots(1, num_actions, figsize=(5*num_actions, 4))
        fig.suptitle("Q-value Heatmaps by Action", fontsize=16)
        
        # Define action names for subplot titles
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
            for s in range(self.model.num_states):
                if s != self.model.fictional_end_state:  # Skip fictional end state
                    # Convert state index to grid coordinates
                    row, col = self.model.state2cell(s)
                    q_values[row, col] = self.Q[s, a]
            
            # Create mask for cells that aren't valid states
            mask = np.ones_like(q_values, dtype=bool)
            for s in range(self.model.num_states):
                if s != self.model.fictional_end_state:
                    row, col = self.model.state2cell(s)
                    mask[row, col] = False
            
            # Plot the heatmap
            ax = axes[a]
            cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
            im = sns.heatmap(q_values, 
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
            start_row, start_col = self.model.state2cell(self.model.start_state)
            goal_row, goal_col = self.model.state2cell(self.model.goal_state)
            
            # Draw rectangles around special cells
            ax.add_patch(plt.Rectangle((start_col, start_row), 1, 1, fill=False, 
                                     edgecolor='blue', linewidth=3, label='Start'))
            ax.add_patch(plt.Rectangle((goal_col, goal_row), 1, 1, fill=False, 
                                     edgecolor='green', linewidth=3, label='Goal'))
            
            # Set title and labels
            ax.set_title(f"Action: {action_names[Actions(a)]}")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig


def run_sarsa_comparison(
    model: Model, 
    num_episodes: int = 1000, 
    max_steps: int = 1000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    epsilon_decay: Optional[float] = 0.001,
    use_expected_sarsa: bool = True
):
    """
    Run and compare SARSA and Expected SARSA on the same environment.
    
    Args:
        model: The environment model
        num_episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        alpha: Learning rate
        epsilon: Exploration rate
        epsilon_decay: Epsilon decay rate
        use_expected_sarsa: Whether to run Expected SARSA comparison
        
    Returns:
        results: Dictionary with results and statistics
    """
    print(f"Running SARSA comparison on {model.__class__.__name__}")
    print(f"Parameters: alpha={alpha}, epsilon={epsilon}, epsilon_decay={epsilon_decay}")
    print(f"Training for {num_episodes} episodes with max {max_steps} steps per episode")
    
    # Create SARSA agent
    sarsa_agent = SarsaAgent(
        model=model,
        alpha=alpha,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay
    )
    
    # Train SARSA
    print("\nTraining SARSA...")
    sarsa_policy, sarsa_q, sarsa_stats = sarsa_agent.train(
        num_episodes=num_episodes,
        max_steps=max_steps,
        expected_sarsa=False,
        verbose=False
    )
    
    # Create value function from Q-values
    sarsa_V = np.max(sarsa_q, axis=1)
    
    results = {
        'SARSA': {
            'policy': sarsa_policy,
            'Q': sarsa_q,
            'V': sarsa_V,
            'stats': sarsa_stats
        }
    }
    
    # Plot SARSA learning
    sarsa_agent.visualize_learning()
    plt.title("SARSA Learning Progress")
    plt.savefig("sarsa_learning.png")
    
    # Plot steps per episode
    sarsa_agent.plot_steps_per_episode()
    plt.title("SARSA Steps per Episode")
    plt.savefig("sarsa_steps_per_episode.png")
    
    # Plot Q-value heatmaps
    sarsa_agent.plot_q_value_heatmaps()
    plt.title("SARSA Q-value Heatmaps")
    plt.savefig("sarsa_q_heatmaps.png")
    
    if use_expected_sarsa:
        # Create Expected SARSA agent
        expected_sarsa_agent = SarsaAgent(
            model=model,
            alpha=alpha,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay
        )
        
        # Train Expected SARSA
        print("\nTraining Expected SARSA...")
        expected_sarsa_policy, expected_sarsa_q, expected_sarsa_stats = expected_sarsa_agent.train(
            num_episodes=num_episodes,
            max_steps=max_steps,
            expected_sarsa=True,
            verbose=False
        )
        
        # Create value function from Q-values
        expected_sarsa_V = np.max(expected_sarsa_q, axis=1)
        
        results['Expected SARSA'] = {
            'policy': expected_sarsa_policy,
            'Q': expected_sarsa_q,
            'V': expected_sarsa_V,
            'stats': expected_sarsa_stats
        }
        
        # Plot Expected SARSA learning
        expected_sarsa_agent.visualize_learning()
        plt.title("Expected SARSA Learning Progress")
        plt.savefig("expected_sarsa_learning.png")
        
        # Plot steps per episode for Expected SARSA
        expected_sarsa_agent.plot_steps_per_episode()
        plt.title("Expected SARSA Steps per Episode")
        plt.savefig("expected_sarsa_steps_per_episode.png")
        
        # Plot Q-value heatmaps for Expected SARSA
        expected_sarsa_agent.plot_q_value_heatmaps()
        plt.title("Expected SARSA Q-value Heatmaps")
        plt.savefig("expected_sarsa_q_heatmaps.png")
        
        # Compare learning curves
        plt.figure(figsize=(12, 8))
        
        # Plot episode returns
        plt.subplot(2, 1, 1)
        plt.plot(sarsa_stats['episode_returns'], label='SARSA')
        plt.plot(expected_sarsa_stats['episode_returns'], label='Expected SARSA')
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Episode Returns")
        plt.legend()
        
        # Smooth returns for better visualization
        window_size = min(50, len(sarsa_stats['episode_returns']))
        smoothed_sarsa = np.convolve(
            sarsa_stats['episode_returns'], 
            np.ones(window_size) / window_size, 
            mode='valid'
        )
        smoothed_expected = np.convolve(
            expected_sarsa_stats['episode_returns'], 
            np.ones(window_size) / window_size, 
            mode='valid'
        )
        
        plt.subplot(2, 1, 2)
        plt.plot(smoothed_sarsa, label='SARSA')
        plt.plot(smoothed_expected, label='Expected SARSA')
        plt.xlabel("Episode")
        plt.ylabel("Smoothed Return")
        plt.title(f"Smoothed Returns (window={window_size})")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("sarsa_comparison.png")
    
    # Create results directory if it doesn't exist
    results_dir = "/Users/marinafranca/Desktop/gridworld-rl/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Compare policies and value functions - only showing heatmaps now
    if use_expected_sarsa:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle("SARSA vs Expected SARSA - Final Policies and Value Functions", fontsize=16)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 7))
        fig.suptitle("SARSA - Final Policy and Value Function", fontsize=16)
        axes = [axes]  # Make it a list for consistent indexing
    
    # Prepare value functions for plotting
    def prepare_value_func_for_plot(V):
        V_copy = V.copy()
        V_copy[np.isneginf(V_copy)] = -1000  # Replace -inf with large negative number
        return V_copy
    
    # Convert policies from indices to Actions enum
    sarsa_policy_enum = np.array([Actions(a) for a in sarsa_policy])
    
    # Plot SARSA policy and value function
    plot_vp(model, prepare_value_func_for_plot(sarsa_V), sarsa_policy_enum, ax=axes[0])
    axes[0].set_title("SARSA: Value Function & Policy")
    
    # Save textual results to a file instead of plotting them
    results_file = os.path.join(results_dir, "sarsa_comparison_results.txt")
    with open(results_file, 'w') as f:
        f.write("===== SARSA vs Expected SARSA Summary =====\n")
        f.write(f"SARSA training time: {sarsa_stats['training_time']:.2f}s\n")
        
        if use_expected_sarsa:
            # Convert Expected SARSA policy from indices to Actions enum
            expected_sarsa_policy_enum = np.array([Actions(a) for a in expected_sarsa_policy])
            
            # Plot Expected SARSA policy and value function
            plot_vp(model, prepare_value_func_for_plot(expected_sarsa_V), expected_sarsa_policy_enum, ax=axes[1])
            axes[1].set_title("Expected SARSA: Value Function & Policy")
            
            # Add Expected SARSA results to file
            f.write(f"Expected SARSA training time: {expected_sarsa_stats['training_time']:.2f}s\n")
            f.write(f"Time difference: {abs(sarsa_stats['training_time'] - expected_sarsa_stats['training_time']):.2f}s "
                  f"({'Expected SARSA' if expected_sarsa_stats['training_time'] < sarsa_stats['training_time'] else 'SARSA'} was faster)\n")
            
            sarsa_avg_return = np.mean(sarsa_stats['episode_returns'][-100:])
            expected_avg_return = np.mean(expected_sarsa_stats['episode_returns'][-100:])
            f.write(f"SARSA average return (last 100 episodes): {sarsa_avg_return:.2f}\n")
            f.write(f"Expected SARSA average return (last 100 episodes): {expected_avg_return:.2f}\n")
            f.write(f"Return difference: {abs(sarsa_avg_return - expected_avg_return):.2f} "
                  f"({'Expected SARSA' if expected_avg_return > sarsa_avg_return else 'SARSA'} was better)\n")
            
            policy_diff = np.sum(sarsa_policy != expected_sarsa_policy)
            policy_diff_pct = policy_diff / len(sarsa_policy) * 100
            f.write(f"Policy differences: {policy_diff} states ({policy_diff_pct:.2f}%)\n")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("sarsa_final_policies.png")
    
    # Print a message about where results are saved
    print(f"\nDetailed comparison results saved to: {results_file}")
    
    return results


def hyperparameter_tuning_demo(model: Model):
    """
    Demonstrate the effect of different hyperparameters on SARSA learning using heatmaps.
    
    Creates a 3×3 grid of heatmaps, each showing final average returns for different
    combinations of alpha and epsilon, for a specific (num_episodes, max_steps) pair.
    
    Args:
        model: The environment model
        
    Returns:
        best_performance: Dictionary with best hyperparameters
    """
    # Define the hyperparameter values to test
    alphas = [0.01, 0.1, 0.5]
    epsilons = [0.01, 0.1, 0.3]
    num_episodes_values = [100, 500, 1000]
    max_steps_values = [50, 200, 500]
    
    # Initialize a 4D array to store results
    # Dimensions: [num_episodes_idx, max_steps_idx, alpha_idx, epsilon_idx]
    results = np.zeros((len(num_episodes_values), len(max_steps_values), len(alphas), len(epsilons)))
    
    # Track best performance for summary
    best_performance = {
        'alpha': None,
        'epsilon': None,
        'num_episodes': None,
        'max_steps': None,
        'return': -float('inf')
    }
    
    # Systematically iterate over all hyperparameter combinations
    for e_idx, num_episodes in enumerate(num_episodes_values):
        for s_idx, max_steps in enumerate(max_steps_values):
            for a_idx, alpha in enumerate(alphas):
                for eps_idx, epsilon in enumerate(epsilons):
                    # Print progress update
                    print(f"\nTesting SARSA with α={alpha}, ε={epsilon}, " +
                          f"episodes={num_episodes}, max_steps={max_steps}")
                    
                    # Create and train agent with this specific parameter combination
                    agent = SarsaAgent(
                        model=model,
                        alpha=alpha,
                        epsilon=epsilon
                    )
                    
                    # Train with these parameters
                    _, _, stats = agent.train(
                        num_episodes=num_episodes,
                        max_steps=max_steps,
                        verbose=False
                    )
                    
                    # Calculate final performance (average of last 50 episodes or 20% of episodes, whichever is smaller)
                    final_window = min(50, max(10, int(num_episodes * 0.2)))
                    final_return_avg = np.mean(stats['episode_returns'][-final_window:])
                    
                    # Store the result
                    results[e_idx, s_idx, a_idx, eps_idx] = final_return_avg
                    
                    # Check if this is the best performance so far
                    if final_return_avg > best_performance['return']:
                        best_performance['alpha'] = alpha
                        best_performance['epsilon'] = epsilon
                        best_performance['num_episodes'] = num_episodes
                        best_performance['max_steps'] = max_steps
                        best_performance['return'] = final_return_avg
    
    # Create a figure with a grid of heatmaps
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle("SARSA Hyperparameter Tuning: Final Average Returns", fontsize=20)
    
    # Find global min and max for consistent color scaling across all heatmaps
    vmin = np.min(results)
    vmax = np.max(results)
    
    # Create heatmaps for each (num_episodes, max_steps) combination
    for e_idx, num_episodes in enumerate(num_episodes_values):
        for s_idx, max_steps in enumerate(max_steps_values):
            # Get the current subplot
            ax = axes[e_idx, s_idx]
            
            # Create heatmap for this (num_episodes, max_steps) combination
            heatmap_data = results[e_idx, s_idx]
            
            # Create heatmap with seaborn - no per-subplot color bars
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
                cbar=False  # No individual color bars
            )
            
            # Set title and labels
            ax.set_title(f"Episodes={num_episodes}, Steps={max_steps}", fontsize=12)
            ax.set_xlabel("Epsilon (ε)" if e_idx == 2 else "")  # Only add xlabel on bottom row
            ax.set_ylabel("Alpha (α)" if s_idx == 0 else "")    # Only add ylabel on leftmost column
            
            # Mark the best parameter combination if it's in this heatmap
            if (best_performance['num_episodes'] == num_episodes and 
                best_performance['max_steps'] == max_steps):
                best_a_idx = alphas.index(best_performance['alpha'])
                best_eps_idx = epsilons.index(best_performance['epsilon'])
                ax.add_patch(plt.Rectangle(
                    (best_eps_idx, best_a_idx), 1, 1, 
                    fill=False, edgecolor='red', linewidth=3, clip_on=False
                ))
    
    # Add a common colorbar for all heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Final Average Return', rotation=270, labelpad=20)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Create results directory if it doesn't exist
    results_dir = "/Users/marinafranca/Desktop/gridworld-rl/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Write best combination information to file instead of plotting it
    results_file = os.path.join(results_dir, "sarsa_comparison_results.txt")
    with open(results_file, 'a') as f:  # Append mode so we don't overwrite existing content
        f.write("\n\n===== SARSA Hyperparameter Tuning Results =====\n")
        f.write(f"Best parameter combination:\n")
        f.write(f"  Learning rate (α): {best_performance['alpha']}\n")
        f.write(f"  Exploration rate (ε): {best_performance['epsilon']}\n")
        f.write(f"  Number of episodes: {best_performance['num_episodes']}\n")
        f.write(f"  Maximum steps: {best_performance['max_steps']}\n")
        f.write(f"  Average return: {best_performance['return']:.2f}\n")
    
    # Save the visualization
    plt.savefig("sarsa_hyperparameter_heatmaps.png", dpi=300, bbox_inches='tight')
    
    # Print best hyperparameter combination to console
    print("\n===== Hyperparameter Tuning Results =====")
    print(f"Best parameter combination:")
    print(f"  Learning rate (α): {best_performance['alpha']}")
    print(f"  Exploration rate (ε): {best_performance['epsilon']}")
    print(f"  Number of episodes: {best_performance['num_episodes']}")
    print(f"  Maximum steps: {best_performance['max_steps']}")
    print(f"  Average return: {best_performance['return']:.2f}")
    print(f"\nResults saved to: {results_file}")
    
    # General parameter guidance
    print("\nGeneral parameter guidelines:")
    print("  α: Start around 0.1, potentially with decay over episodes")
    print("  ε: Start around 0.1-0.2, with decay to ensure eventual convergence")
    print("  Episodes: More is generally better, but diminishing returns after convergence")
    print("  Max steps: Should be large enough to allow goal completion, but not wastefully large")
    
    return best_performance


if __name__ == "__main__":
    from environment.world_config import small_world
    from environment.model import Model
    
    # Create model
    model = Model(small_world)
    
    print("===== SARSA and Expected SARSA Implementation =====")
    print("World: small_world")
    print(f"Size: {small_world.num_rows}x{small_world.num_cols}")
    print(f"Discount factor (gamma): {model.gamma}")
    
    # Run hyperparameter tuning demo to find optimal parameters
    print("\n===== Running Hyperparameter Tuning =====")
    best_params = hyperparameter_tuning_demo(model)
    
    # Extract the optimal parameters
    optimal_alpha = best_params['alpha']
    optimal_epsilon = best_params['epsi lon']
    optimal_episodes = best_params['num_episodes']
    optimal_max_steps = best_params['max_steps']
    
    print(f"\n===== Using Optimal Hyperparameters from Tuning =====")
    print(f"Learning rate (α): {optimal_alpha}")
    print(f"Exploration rate (ε): {optimal_epsilon}")
    print(f"Number of episodes: {optimal_episodes}")
    print(f"Maximum steps: {optimal_max_steps}")
    
    # Run comparison with optimal hyperparameters
    print("\n===== Running SARSA and Expected SARSA with Optimal Hyperparameters =====")
    results = run_sarsa_comparison(
        model=model,
        num_episodes=optimal_episodes,
        max_steps=optimal_max_steps,
        alpha=optimal_alpha,
        epsilon=optimal_epsilon,
        epsilon_decay=0.001  # Keep epsilon decay - could make this tunable too if desired
    )
    
    # Create results directory if it doesn't exist
    results_dir = "/Users/marinafranca/Desktop/gridworld-rl/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the optimal parameters to a separate file for reference
    optimal_params_file = os.path.join(results_dir, "optimal_sarsa_parameters.txt")
    with open(optimal_params_file, 'w') as f:
        f.write("===== Optimal SARSA Parameters =====\n")
        f.write(f"Learning rate (α): {optimal_alpha}\n")
        f.write(f"Exploration rate (ε): {optimal_epsilon}\n")
        f.write(f"Number of episodes: {optimal_episodes}\n")
        f.write(f"Maximum steps: {optimal_max_steps}\n")
        f.write(f"Average return: {best_params['return']:.2f}\n")
        f.write("\nThese parameters were determined by a grid search over 81 combinations of hyperparameters.\n")
        f.write("They were automatically applied to all subsequent SARSA and Expected SARSA experiments.\n")
    
    print(f"Optimal parameters saved to: {optimal_params_file}")
    
    plt.show() 