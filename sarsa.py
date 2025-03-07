import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from model import Model, Actions
from plot_vp import plot_vp
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
    
    # Compare policies and value functions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("SARSA vs Expected SARSA - Final Policies and Value Functions", fontsize=16)
    
    # Prepare value functions for plotting
    def prepare_value_func_for_plot(V):
        V_copy = V.copy()
        V_copy[np.isneginf(V_copy)] = -1000  # Replace -inf with large negative number
        return V_copy
    
    # Convert policies from indices to Actions enum
    sarsa_policy_enum = np.array([Actions(a) for a in sarsa_policy])
    
    # Plot SARSA policy and value function
    plot_vp(model, prepare_value_func_for_plot(sarsa_V), sarsa_policy_enum, ax=axes[0, 0])
    axes[0, 0].set_title("SARSA: Value Function & Policy")
    
    if use_expected_sarsa:
        # Convert Expected SARSA policy from indices to Actions enum
        expected_sarsa_policy_enum = np.array([Actions(a) for a in expected_sarsa_policy])
        
        # Plot Expected SARSA policy and value function
        plot_vp(model, prepare_value_func_for_plot(expected_sarsa_V), expected_sarsa_policy_enum, ax=axes[0, 1])
        axes[0, 1].set_title("Expected SARSA: Value Function & Policy")
        
        # Policy difference
        policy_diff = np.sum(sarsa_policy != expected_sarsa_policy)
        axes[1, 0].text(0.5, 0.5, f"Policy differences: {policy_diff} states", 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=14)
        axes[1, 0].axis('off')
        
        # Performance comparison
        sarsa_time = sarsa_stats['training_time']
        expected_time = expected_sarsa_stats['training_time']
        sarsa_avg_return = np.mean(sarsa_stats['episode_returns'][-100:])
        expected_avg_return = np.mean(expected_sarsa_stats['episode_returns'][-100:])
        
        comparison_text = (
            f"Performance Comparison:\n\n"
            f"SARSA:\n"
            f"  Training time: {sarsa_time:.2f}s\n"
            f"  Avg return (last 100 episodes): {sarsa_avg_return:.2f}\n\n"
            f"Expected SARSA:\n"
            f"  Training time: {expected_time:.2f}s\n"
            f"  Avg return (last 100 episodes): {expected_avg_return:.2f}\n\n"
            f"{'Expected SARSA' if expected_avg_return > sarsa_avg_return else 'SARSA'} "
            f"achieved better performance.\n"
            f"{'Expected SARSA' if expected_time < sarsa_time else 'SARSA'} "
            f"was faster to train."
        )
        
        axes[1, 1].text(0.5, 0.5, comparison_text, 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=12)
        axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("sarsa_final_policies.png")
    
    # Print final summary
    print("\n===== SARSA vs Expected SARSA Summary =====")
    print(f"SARSA training time: {sarsa_stats['training_time']:.2f}s")
    if use_expected_sarsa:
        print(f"Expected SARSA training time: {expected_sarsa_stats['training_time']:.2f}s")
        print(f"Time difference: {abs(sarsa_stats['training_time'] - expected_sarsa_stats['training_time']):.2f}s "
              f"({'Expected SARSA' if expected_sarsa_stats['training_time'] < sarsa_stats['training_time'] else 'SARSA'} was faster)")
        
        sarsa_avg_return = np.mean(sarsa_stats['episode_returns'][-100:])
        expected_avg_return = np.mean(expected_sarsa_stats['episode_returns'][-100:])
        print(f"SARSA average return (last 100 episodes): {sarsa_avg_return:.2f}")
        print(f"Expected SARSA average return (last 100 episodes): {expected_avg_return:.2f}")
        print(f"Return difference: {abs(sarsa_avg_return - expected_avg_return):.2f} "
              f"({'Expected SARSA' if expected_avg_return > sarsa_avg_return else 'SARSA'} was better)")
        
        policy_diff = np.sum(sarsa_policy != expected_sarsa_policy)
        policy_diff_pct = policy_diff / len(sarsa_policy) * 100
        print(f"Policy differences: {policy_diff} states ({policy_diff_pct:.2f}%)")
    
    return results


def hyperparameter_tuning_demo(model: Model):
    """
    Demonstrate the effect of different hyperparameters on SARSA learning.
    
    Args:
        model: The environment model
    """
    alphas = [0.01, 0.1, 0.5]
    epsilons = [0.01, 0.1, 0.3]
    
    fig, axes = plt.subplots(len(alphas), len(epsilons), figsize=(15, 10))
    fig.suptitle("SARSA Hyperparameter Tuning: Impact of α and ε", fontsize=16)
    
    # Run SARSA with different hyperparameters
    for i, alpha in enumerate(alphas):
        for j, epsilon in enumerate(epsilons):
            print(f"\nTesting SARSA with alpha={alpha}, epsilon={epsilon}")
            
            # Create agent
            agent = SarsaAgent(
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
            ax.set_title(f"α={alpha}, ε={epsilon}")
            
            if i == len(alphas) - 1:
                ax.set_xlabel("Episode")
            if j == 0:
                ax.set_ylabel("Smoothed Return")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("sarsa_hyperparameter_tuning.png")
    
    # Discussion of results
    print("\n===== Hyperparameter Tuning Results =====")
    print("Learning rate (α):")
    print("  Too small (0.01): Learning is very slow, requires many episodes to converge")
    print("  Moderate (0.1): Good balance between learning speed and stability")
    print("  Too large (0.5): Can cause oscillations or divergence in Q-values")
    
    print("\nExploration rate (ε):")
    print("  Too small (0.01): May get stuck in suboptimal policies (not enough exploration)")
    print("  Moderate (0.1): Good balance between exploration and exploitation")
    print("  Too large (0.3): Too much random exploration, slower convergence to optimal policy")
    
    print("\nRecommended values:")
    print("  α: Start around 0.1, potentially with decay over episodes")
    print("  ε: Start around 0.1-0.2, with decay to ensure eventual convergence to deterministic policy")


if __name__ == "__main__":
    from world_config import small_world
    from model import Model
    
    # Create model
    model = Model(small_world)
    
    print("===== SARSA and Expected SARSA Implementation =====")
    print("World: small_world")
    print(f"Size: {small_world.num_rows}x{small_world.num_cols}")
    print(f"Discount factor (gamma): {model.gamma}")
    
    # Run hyperparameter tuning demo
    hyperparameter_tuning_demo(model)
    
    # Run comparison with selected hyperparameters
    print("\n===== Running SARSA and Expected SARSA with selected hyperparameters =====")
    results = run_sarsa_comparison(
        model=model,
        num_episodes=500,  # More episodes for better convergence
        max_steps=1000,
        alpha=0.1,
        epsilon=0.1,
        epsilon_decay=0.001
    )
    
    plt.show() 