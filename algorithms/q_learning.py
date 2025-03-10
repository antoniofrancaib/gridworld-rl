import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.model import Model, Actions
from utils.plot_vp import plot_vp
from typing import List, Tuple, Dict, Optional, Callable
from sarsa import SarsaAgent


class QLearningAgent:
    """
    Implementation of Q-Learning algorithm.
    
    Q-Learning is an off-policy TD control algorithm that learns the action-value 
    function Q(s,a) and updates it based on the maximum Q-value in the next state,
    regardless of which action is actually taken.
    
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
        Initialize the Q-Learning agent.
        
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
            action: Selected action (as integer)
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(len(Actions))
        else:
            # Exploit: best action according to Q-values
            return np.argmax(self.Q[state])
    
    def q_learning_update(self, s: int, a: int, r: float, s_next: int) -> float:
        """
        Perform a Q-learning update.
        
        Args:
            s: Current state
            a: Current action
            r: Reward received
            s_next: Next state
            
        Returns:
            delta: TD error (change in Q-value)
        """
        # Q-learning update rule: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        current_q = self.Q[s, a]
        
        # Get maximum Q-value for next state - this is the key difference from SARSA
        max_next_q = np.max(self.Q[s_next]) if s_next != self.model.fictional_end_state else 0
        
        # TD target: r + γ max_a' Q(s',a')
        td_target = r + self.gamma * max_next_q
        
        # TD error: r + γ max_a' Q(s',a') - Q(s,a)
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
        verbose: bool = True,
        early_stopping: Optional[Dict] = None
    ):
        """
        Train the agent using Q-learning.
        
        Args:
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
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
        if verbose:
            print(f"Training Q-learning for {num_episodes} episodes (max {max_steps} steps per episode)")
            print(f"  alpha={self.alpha_init} {'with decay' if self.alpha_decay else 'fixed'}")
            print(f"  epsilon={self.epsilon_init} {'with decay' if self.epsilon_decay else 'fixed'}")
            print(f"  gamma={self.gamma}")
        
        start_time = time.time()
        
        for episode in tqdm(range(num_episodes), desc=f"Training Q-learning"):
            # Reset agent state
            s = self.model.start_state  # Start at initial state
            
            # Track episode statistics
            episode_return = 0
            step = 0
            
            # Run episode
            done = False
            while not done and step < max_steps:
                # Select action using epsilon-greedy policy
                a = self.epsilon_greedy_policy(s)
                
                # Sample next state based on transition probabilities
                p_s_next = [
                    self.model.transition_probability(s, s_next_candidate, Actions(a))
                    for s_next_candidate in range(self.model.num_states)
                ]
                
                # Sample next state
                s_next = np.random.choice(self.model.num_states, p=p_s_next)
                
                # Get reward
                r = self.model.reward(s, Actions(a))
                
                # Check if terminal state or return-to-start from cliff
                if s_next == self.model.fictional_end_state or s_next == self.model.goal_state:
                    done = True
                # Special handling for cliff world: if we're back at start after being in a bad state
                elif s in self.model.bad_states and s_next == self.model.start_state and self.model.world.return_to_start_from_bad_state:
                    # We fell off the cliff and returned to start - don't end episode
                    pass
                
                # Update Q-values using Q-learning (different from SARSA)
                delta = self.q_learning_update(s, a, r, s_next)
                
                # Move to next state
                s = s_next
                
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
    
    def visualize_learning(self, title="Q-Learning Progress"):
        """
        Visualize the learning progress.
        
        Args:
            title: Title for the plot
        
        Returns:
            fig: The generated figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
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


def run_q_learning_vs_sarsa_comparison(
    model: Model, 
    num_episodes: int = 1000, 
    max_steps: int = 1000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    epsilon_decay: Optional[float] = 0.001,
    plot_save_name: str = "q_learning_vs_sarsa"
):
    """
    Run and compare Q-learning and SARSA on the same environment.
    
    Args:
        model: The environment model
        num_episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        alpha: Learning rate
        epsilon: Exploration rate
        epsilon_decay: Epsilon decay rate
        plot_save_name: Base name for saving plots
        
    Returns:
        results: Dictionary with results and statistics
    """
    print(f"Running Q-learning vs SARSA comparison on {model.__class__.__name__}")
    print(f"Parameters: alpha={alpha}, epsilon={epsilon}, epsilon_decay={epsilon_decay}")
    print(f"Training for {num_episodes} episodes with max {max_steps} steps per episode")
    
    # Create Q-learning agent
    q_learning_agent = QLearningAgent(
        model=model,
        alpha=alpha,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay
    )
    
    # Train Q-learning
    print("\nTraining Q-learning...")
    q_learning_policy, q_learning_q, q_learning_stats = q_learning_agent.train(
        num_episodes=num_episodes,
        max_steps=max_steps,
        verbose=True
    )
    
    # Create value function from Q-values
    q_learning_V = np.max(q_learning_q, axis=1)
    
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
        verbose=True
    )
    
    # Create value function from Q-values
    sarsa_V = np.max(sarsa_q, axis=1)
    
    results = {
        'Q-learning': {
            'policy': q_learning_policy,
            'Q': q_learning_q,
            'V': q_learning_V,
            'stats': q_learning_stats
        },
        'SARSA': {
            'policy': sarsa_policy,
            'Q': sarsa_q,
            'V': sarsa_V,
            'stats': sarsa_stats
        }
    }
    
    # Plot individual learning curves
    q_learning_agent.visualize_learning("Q-Learning Progress")
    plt.savefig(f"{plot_save_name}_q_learning.png", dpi=300, bbox_inches='tight')
    
    # SarsaAgent's visualize_learning doesn't accept a title parameter
    sarsa_agent.visualize_learning()
    plt.title("SARSA Learning Progress")  # Add title directly to the plot
    plt.savefig(f"{plot_save_name}_sarsa.png", dpi=300, bbox_inches='tight')
    
    # Compare learning curves
    plt.figure(figsize=(12, 8))
    
    # Plot episode returns
    plt.subplot(2, 1, 1)
    plt.plot(q_learning_stats['episode_returns'], label='Q-learning')
    plt.plot(sarsa_stats['episode_returns'], label='SARSA')
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode Returns")
    plt.legend()
    
    # Smooth returns for better visualization
    window_size = min(50, len(q_learning_stats['episode_returns']))
    smoothed_q_learning = np.convolve(
        q_learning_stats['episode_returns'], 
        np.ones(window_size) / window_size, 
        mode='valid'
    )
    smoothed_sarsa = np.convolve(
        sarsa_stats['episode_returns'], 
        np.ones(window_size) / window_size, 
        mode='valid'
    )
    
    plt.subplot(2, 1, 2)
    plt.plot(smoothed_q_learning, label='Q-learning')
    plt.plot(smoothed_sarsa, label='SARSA')
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Return")
    plt.title(f"Smoothed Returns (window={window_size})")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{plot_save_name}_learning_curves.png", dpi=300, bbox_inches='tight')
    
    # Compare policies and value functions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Q-learning vs SARSA - Final Policies and Value Functions", fontsize=16)
    
    # Prepare value functions for plotting
    def prepare_value_func_for_plot(V):
        V_copy = V.copy()
        V_copy[np.isneginf(V_copy)] = -1000  # Replace -inf with large negative number
        return V_copy
    
    # Convert policies from indices to Actions enum
    q_learning_policy_enum = np.array([Actions(a) for a in q_learning_policy])
    sarsa_policy_enum = np.array([Actions(a) for a in sarsa_policy])
    
    # Plot Q-learning policy and value function
    plot_vp(model, prepare_value_func_for_plot(q_learning_V), q_learning_policy_enum, ax=axes[0, 0])
    axes[0, 0].set_title("Q-learning: Value Function & Policy")
    
    # Plot SARSA policy and value function
    plot_vp(model, prepare_value_func_for_plot(sarsa_V), sarsa_policy_enum, ax=axes[0, 1])
    axes[0, 1].set_title("SARSA: Value Function & Policy")
    
    # Policy difference
    policy_diff = np.sum(q_learning_policy != sarsa_policy)
    policy_diff_pct = policy_diff / len(q_learning_policy) * 100
    different_states_text = f"Policy differences: {policy_diff} states ({policy_diff_pct:.2f}%)\n\n"
    
    if policy_diff > 0:
        different_states_text += "States with different actions:\n"
        count = 0
        for s in range(model.num_states):
            if q_learning_policy[s] != sarsa_policy[s]:
                if count < 5:  # Show at most 5 examples
                    cell = model.state2cell(s)
                    different_states_text += f"State {s} (Cell {cell}):\n"
                    different_states_text += f"  Q-learning: {Actions(q_learning_policy[s])}\n"
                    different_states_text += f"  SARSA: {Actions(sarsa_policy[s])}\n"
                count += 1
        if count > 5:
            different_states_text += f"...and {count - 5} more states\n"
    
    axes[1, 0].text(0.5, 0.5, different_states_text,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12)
    axes[1, 0].axis('off')
    
    # Performance comparison
    q_learning_time = q_learning_stats['training_time']
    sarsa_time = sarsa_stats['training_time']
    q_learning_avg_return = np.mean(q_learning_stats['episode_returns'][-100:])
    sarsa_avg_return = np.mean(sarsa_stats['episode_returns'][-100:])
    
    comparison_text = (
        f"Performance Comparison:\n\n"
        f"Q-learning:\n"
        f"  Training time: {q_learning_time:.2f}s\n"
        f"  Avg return (last 100 episodes): {q_learning_avg_return:.2f}\n\n"
        f"SARSA:\n"
        f"  Training time: {sarsa_time:.2f}s\n"
        f"  Avg return (last 100 episodes): {sarsa_avg_return:.2f}\n\n"
        f"{'Q-learning' if q_learning_avg_return > sarsa_avg_return else 'SARSA'} "
        f"achieved better performance.\n"
        f"{'Q-learning' if q_learning_time < sarsa_time else 'SARSA'} "
        f"was faster to train."
    )
    
    axes[1, 1].text(0.5, 0.5, comparison_text,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{plot_save_name}_final_policies.png", dpi=300, bbox_inches='tight')
    
    # Print final summary
    print("\n===== Q-learning vs SARSA Summary =====")
    print(f"Q-learning training time: {q_learning_time:.2f}s")
    print(f"SARSA training time: {sarsa_time:.2f}s")
    print(f"Time difference: {abs(q_learning_time - sarsa_time):.2f}s "
          f"({'Q-learning' if q_learning_time < sarsa_time else 'SARSA'} was faster)")
    
    print(f"Q-learning average return (last 100 episodes): {q_learning_avg_return:.2f}")
    print(f"SARSA average return (last 100 episodes): {sarsa_avg_return:.2f}")
    print(f"Return difference: {abs(q_learning_avg_return - sarsa_avg_return):.2f} "
          f"({'Q-learning' if q_learning_avg_return > sarsa_avg_return else 'SARSA'} was better)")
    
    print(f"Policy differences: {policy_diff} states ({policy_diff_pct:.2f}%)")
    
    print("\nTheoretical Explanation of Differences:")
    print("1. Q-learning is an off-policy algorithm that learns the optimal policy")
    print("   directly, while SARSA is an on-policy algorithm that learns the policy")
    print("   it's following (including exploration).")
    print("2. Q-learning tends to find more optimal policies faster but may be less")
    print("   stable during learning due to its maximization step.")
    print("3. SARSA tends to be more conservative in risky situations because it")
    print("   accounts for the exploration policy it's using.")
    print("4. In environments with high stochasticity or penalty states near optimal")
    print("   paths, SARSA may learn safer policies that avoid catastrophic failures.")
    
    print("\nHyperparameter Analysis:")
    print(f"Alpha (learning rate): {alpha}")
    print(f"  - Higher values can accelerate learning but may cause instability")
    print(f"  - Lower values provide more stable but slower learning")
    print(f"Epsilon (exploration rate): {epsilon} {'with decay' if epsilon_decay else 'fixed'}")
    print(f"  - Higher values promote more exploration but slower convergence")
    print(f"  - Lower values exploit current knowledge but might miss optimal policy")
    print(f"  - Decay schedule allows initial exploration with eventual exploitation")
    
    return results


def hyperparameter_tuning_q_learning(model: Model):
    """
    Demonstrate the effect of different hyperparameters on Q-learning.
    
    Args:
        model: The environment model
    """
    alphas = [0.01, 0.1, 0.5]
    epsilons = [0.01, 0.1, 0.3]
    
    fig, axes = plt.subplots(len(alphas), len(epsilons), figsize=(15, 10))
    fig.suptitle("Q-learning Hyperparameter Tuning: Impact of α and ε", fontsize=16)
    
    best_return = -float('inf')
    best_params = None
    
    # Run Q-learning with different hyperparameters
    for i, alpha in enumerate(alphas):
        for j, epsilon in enumerate(epsilons):
            print(f"\nTesting Q-learning with alpha={alpha}, epsilon={epsilon}")
            
            # Create agent
            agent = QLearningAgent(
                model=model,
                alpha=alpha,
                epsilon=epsilon
            )
            
            # Train for fewer episodes to speed up demo
            _, _, stats = agent.train(
                num_episodes=200,
                max_steps=500,
                verbose=False
            )
            
            # Track best parameters
            avg_return = np.mean(stats['episode_returns'][-50:])
            if avg_return > best_return:
                best_return = avg_return
                best_params = (alpha, epsilon)
            
            # Plot learning curve
            ax = axes[i, j]
            
            # Smooth returns for better visualization
            window_size = min(20, len(stats['episode_returns']))
            smoothed_returns = np.convolve(
                stats['episode_returns'], 
                np.ones(window_size) / window_size, 
                mode='valid'
            )
            
            ax.plot(smoothed_returns)
            ax.set_title(f"α={alpha}, ε={epsilon}, Avg={avg_return:.2f}")
            
            if i == len(alphas) - 1:
                ax.set_xlabel("Episode")
            if j == 0:
                ax.set_ylabel("Smoothed Return")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("q_learning_hyperparameter_tuning.png", dpi=300, bbox_inches='tight')
    
    # Add text about best parameters
    if best_params:
        best_alpha, best_epsilon = best_params
        plt.figtext(0.5, 0.01, f"Best parameters: α={best_alpha}, ε={best_epsilon}, Avg Return={best_return:.2f}",
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        plt.subplots_adjust(bottom=0.08)
    
    plt.savefig("q_learning_hyperparameter_tuning_with_best.png", dpi=300, bbox_inches='tight')
    
    # Discussion of results
    print("\n===== Q-learning Hyperparameter Tuning Results =====")
    print(f"Best parameters: α={best_params[0]}, ε={best_params[1]}, Avg Return={best_return:.2f}")
    print("\nLearning rate (α):")
    print("  Too small (0.01): Learning is very slow, requires many episodes to converge")
    print("  Moderate (0.1): Good balance between learning speed and stability")
    print("  Too large (0.5): Can cause oscillations or divergence in Q-values")
    
    print("\nExploration rate (ε):")
    print("  Too small (0.01): May get stuck in suboptimal policies (not enough exploration)")
    print("  Moderate (0.1): Good balance between exploration and exploitation")
    print("  Too large (0.3): Too much random exploration, slower convergence to optimal policy")
    
    print("\nRecommended values:")
    print(f"  α: {best_params[0]} (based on our experiments)")
    print(f"  ε: {best_params[1]} (based on our experiments)")
    print("  A decay schedule for epsilon is recommended for final convergence")
    
    return best_params


if __name__ == "__main__":
    from environment.world_config import small_world
    from environment.model import Model
    
    # Create model
    model = Model(small_world)
    
    print("===== Q-learning vs SARSA Implementation and Comparison =====")
    print("World: small_world")
    print(f"Size: {small_world.num_rows}x{small_world.num_cols}")
    print(f"Discount factor (gamma): {model.gamma}")
    
    # Run hyperparameter tuning for Q-learning
    print("\n===== Running Q-learning Hyperparameter Tuning =====")
    best_alpha, best_epsilon = hyperparameter_tuning_q_learning(model)
    
    # Run comparison with best hyperparameters
    print("\n===== Running Q-learning vs SARSA with best hyperparameters =====")
    results = run_q_learning_vs_sarsa_comparison(
        model=model,
        num_episodes=500,  # More episodes for better convergence
        max_steps=1000,
        alpha=best_alpha,
        epsilon=best_epsilon,
        epsilon_decay=0.001
    )
    
    plt.show() 