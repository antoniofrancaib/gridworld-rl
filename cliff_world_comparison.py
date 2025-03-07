import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from world_config import cliff_world
from model import Model, Actions
from q_learning import QLearningAgent
from sarsa import SarsaAgent
from plot_vp import plot_vp

def run_cliff_world_comparison(
    num_episodes=500,
    max_steps=1000,
    alpha=0.1,
    epsilon=0.1,
    epsilon_decay=0.001,
    gamma=None,  # Use environment's default gamma if None
    smooth_window=10,  # Window size for smoothing reward curves
    verbose=True,
    plot_save_name="cliff_world_comparison"
):
    """
    Run a comparison of SARSA and Q-learning on the cliff_world environment.
    
    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        alpha: Learning rate for both algorithms
        epsilon: Initial exploration rate
        epsilon_decay: Decay rate for epsilon
        gamma: Discount factor (if None, use environment default)
        smooth_window: Window size for smoothing reward curves
        verbose: Whether to print progress information
        plot_save_name: Base name for saving plots
        
    Returns:
        Dictionary containing results and statistics
    """
    # Create model for the cliff world environment
    model = Model(cliff_world)
    
    # Override gamma if specified
    if gamma is not None:
        model.gamma = gamma
    
    if verbose:
        print("\n===== Cliff World Environment Details =====")
        print(f"Grid size: {cliff_world.num_rows}x{cliff_world.num_cols}")
        print(f"Start state: {cliff_world.start_cell}, Goal state: {cliff_world.goal_cell}")
        print(f"Cliff cells: {cliff_world.bad_cells}")
        print(f"Rewards: step={cliff_world.reward_step}, goal={cliff_world.reward_goal}, cliff={cliff_world.reward_bad}")
        print(f"Discount factor (gamma): {model.gamma}")
        print(f"Return to start after falling: {cliff_world.return_to_start_from_bad_state}")
        print("\n===== Training Parameters =====")
        print(f"Episodes: {num_episodes}, Max steps per episode: {max_steps}")
        print(f"Alpha (learning rate): {alpha}")
        print(f"Epsilon (exploration): {epsilon} with decay {epsilon_decay}")
    
    # Initialize agents
    sarsa_agent = SarsaAgent(
        model=model,
        alpha=alpha,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay
    )
    
    q_learning_agent = QLearningAgent(
        model=model,
        alpha=alpha,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay
    )
    
    # Train SARSA
    if verbose:
        print("\n===== Training SARSA =====")
    sarsa_start_time = time.time()
    sarsa_policy, sarsa_q, sarsa_stats = sarsa_agent.train(
        num_episodes=num_episodes,
        max_steps=max_steps,
        expected_sarsa=False,  # Use regular SARSA
        verbose=verbose
    )
    sarsa_time = time.time() - sarsa_start_time
    
    # Train Q-learning
    if verbose:
        print("\n===== Training Q-learning =====")
    q_learning_start_time = time.time()
    q_learning_policy, q_learning_q, q_learning_stats = q_learning_agent.train(
        num_episodes=num_episodes,
        max_steps=max_steps,
        verbose=verbose
    )
    q_learning_time = time.time() - q_learning_start_time
    
    # Calculate value functions from Q-values
    sarsa_v = np.max(sarsa_q, axis=1)
    q_learning_v = np.max(q_learning_q, axis=1)
    
    # Prepare data for result analysis
    sarsa_returns = sarsa_stats['episode_returns']
    q_learning_returns = q_learning_stats['episode_returns']
    
    # Compute average returns over the last 50 episodes
    sarsa_final_avg_return = np.mean(sarsa_returns[-50:])
    q_learning_final_avg_return = np.mean(q_learning_returns[-50:])
    
    # Create the Figure 6.13-style plot (Episode vs. Reward)
    plt.figure(figsize=(10, 6))
    
    # Plot raw episode returns
    plt.plot(sarsa_returns, alpha=0.3, color='blue', label='SARSA (raw)')
    plt.plot(q_learning_returns, alpha=0.3, color='red', label='Q-learning (raw)')
    
    # Compute and plot smoothed returns
    if smooth_window > 1:
        # Define a simple moving average function
        def smooth(data, window_size):
            weights = np.ones(window_size) / window_size
            return np.convolve(data, weights, mode='valid')
        
        # Smooth the returns
        smoothed_sarsa = smooth(sarsa_returns, smooth_window)
        smoothed_q = smooth(q_learning_returns, smooth_window)
        
        # Plot smoothed returns (with offset to align with episodes)
        offset = (smooth_window - 1) // 2
        episodes = np.arange(offset, offset + len(smoothed_sarsa))
        plt.plot(episodes, smoothed_sarsa, linewidth=2, color='blue', label=f'SARSA (smoothed, window={smooth_window})')
        plt.plot(episodes, smoothed_q, linewidth=2, color='red', label=f'Q-learning (smoothed, window={smooth_window})')
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards per episode')
    plt.title('Episode Rewards: SARSA vs Q-learning in Cliff World')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{plot_save_name}_rewards.png", dpi=300, bbox_inches='tight')
    
    # Plot policies and value functions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("SARSA vs Q-learning in Cliff World", fontsize=16)
    
    # Convert numeric policies to Actions enum
    sarsa_policy_enum = np.array([Actions(a) for a in sarsa_policy])
    q_learning_policy_enum = np.array([Actions(a) for a in q_learning_policy])
    
    # Prepare value functions for plotting (handle -inf values)
    def prepare_value_func_for_plot(V):
        V_copy = V.copy()
        V_copy[np.isneginf(V_copy)] = -1000  # Replace -inf with large negative number
        return V_copy
    
    # Plot SARSA policy and value function
    plot_vp(model, prepare_value_func_for_plot(sarsa_v), sarsa_policy_enum, ax=axes[0, 0])
    axes[0, 0].set_title(f"SARSA: Final Policy & Value Function\nAvg Return (last 50 episodes): {sarsa_final_avg_return:.2f}")
    
    # Plot Q-learning policy and value function
    plot_vp(model, prepare_value_func_for_plot(q_learning_v), q_learning_policy_enum, ax=axes[0, 1])
    axes[0, 1].set_title(f"Q-learning: Final Policy & Value Function\nAvg Return (last 50 episodes): {q_learning_final_avg_return:.2f}")
    
    # Analyze policy differences
    policy_diff = np.sum(sarsa_policy != q_learning_policy)
    policy_diff_pct = policy_diff / len(sarsa_policy) * 100
    
    # Create text for policy differences
    diff_text = f"Policy Differences: {policy_diff} states ({policy_diff_pct:.2f}%)\n\n"
    if policy_diff > 0:
        diff_text += "States with different actions:\n"
        diff_count = 0
        for s in range(model.num_states):
            if sarsa_policy[s] != q_learning_policy[s] and s != model.fictional_end_state:
                if diff_count < 5:  # Show at most 5 examples
                    cell = model.state2cell(s)
                    diff_text += f"State {s} (Cell {cell}):\n"
                    diff_text += f"  SARSA: {Actions(sarsa_policy[s])}\n"
                    diff_text += f"  Q-learning: {Actions(q_learning_policy[s])}\n"
                diff_count += 1
        if diff_count > 5:
            diff_text += f"...and {diff_count - 5} more states\n"
    
    # Add policy difference text to the plot
    axes[1, 0].text(0.5, 0.5, diff_text,
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=12)
    axes[1, 0].axis('off')
    
    # Create text for performance comparison
    perf_text = (
        f"Performance Comparison:\n\n"
        f"SARSA:\n"
        f"  Training time: {sarsa_time:.2f}s\n"
        f"  Avg return (last 50 episodes): {sarsa_final_avg_return:.2f}\n"
        f"  Min return: {min(sarsa_returns):.2f}\n"
        f"  Max return: {max(sarsa_returns):.2f}\n\n"
        f"Q-learning:\n"
        f"  Training time: {q_learning_time:.2f}s\n"
        f"  Avg return (last 50 episodes): {q_learning_final_avg_return:.2f}\n"
        f"  Min return: {min(q_learning_returns):.2f}\n"
        f"  Max return: {max(q_learning_returns):.2f}\n\n"
        f"Winner: {'SARSA' if sarsa_final_avg_return > q_learning_final_avg_return else 'Q-learning'} "
        f"achieved better final performance."
    )
    
    # Add performance text to the plot
    axes[1, 1].text(0.5, 0.5, perf_text,
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=12)
    axes[1, 1].axis('off')
    
    # Save policy comparison plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{plot_save_name}_policies.png", dpi=300, bbox_inches='tight')
    
    # Print summary
    if verbose:
        print("\n===== Comparison Summary =====")
        print(f"SARSA final avg return (last 50 episodes): {sarsa_final_avg_return:.2f}")
        print(f"Q-learning final avg return (last 50 episodes): {q_learning_final_avg_return:.2f}")
        print(f"Policy differences: {policy_diff} states ({policy_diff_pct:.2f}%)")
        
        print("\n===== Analysis =====")
        print("In the cliff world environment:")
        print(" - SARSA tends to learn a safer policy that stays away from the cliff")
        print("   because it accounts for the exploration during training.")
        print(" - Q-learning tends to learn a more optimal but riskier policy")
        print("   as it learns the optimal deterministic policy directly.")
        print(" - The large negative reward from falling off the cliff influences")
        print("   these algorithms differently - SARSA is more risk-averse because")
        print("   it considers the possibility of random exploratory actions.")
        print(" - Q-learning might have higher variance in episode returns due to")
        print("   occasional falls from the cliff during exploration.")
        
        winner = "SARSA" if sarsa_final_avg_return > q_learning_final_avg_return else "Q-learning"
        print(f"\nOverall, {winner} performed better in terms of final policy performance,")
        print(f"with an average return of {max(sarsa_final_avg_return, q_learning_final_avg_return):.2f}")
        print(f"over the last 50 episodes.")
    
    # Assemble results
    results = {
        'SARSA': {
            'policy': sarsa_policy,
            'Q': sarsa_q,
            'V': sarsa_v,
            'stats': sarsa_stats,
            'training_time': sarsa_time,
            'final_avg_return': sarsa_final_avg_return
        },
        'Q-learning': {
            'policy': q_learning_policy,
            'Q': q_learning_q,
            'V': q_learning_v,
            'stats': q_learning_stats,
            'training_time': q_learning_time,
            'final_avg_return': q_learning_final_avg_return
        },
        'comparison': {
            'policy_diff': policy_diff,
            'policy_diff_pct': policy_diff_pct,
            'winner': 'SARSA' if sarsa_final_avg_return > q_learning_final_avg_return else 'Q-learning'
        }
    }
    
    return results


def analyze_path_safety(model, policy, start_state, goal_state, bad_states, name="Algorithm"):
    """
    Analyze the safety of a policy by checking how close it gets to bad states.
    
    Args:
        model: The environment model
        policy: The policy to analyze
        start_state: Starting state
        goal_state: Goal state
        bad_states: List of bad/cliff states
        name: Name of the algorithm (for display)
        
    Returns:
        Dictionary with analysis results
    """
    # Convert policy to Actions enum
    policy_enum = np.array([Actions(a) for a in policy])
    
    # Simulate deterministic policy execution (no randomness)
    current_state = start_state
    path = [current_state]
    
    step_count = 0
    max_steps = 100  # Prevent infinite loops
    
    min_dist_to_bad = float('inf')
    closest_bad_state = None
    
    while current_state != goal_state and step_count < max_steps:
        # Get action from policy
        action = policy_enum[current_state]
        
        # Find next state deterministically
        next_cell = model._result_action(model.state2cell(current_state), action)
        next_state = model.cell2state(next_cell)
        
        # Add to path
        path.append(next_state)
        current_state = next_state
        step_count += 1
        
        # Check distance to bad states
        for bad_state in bad_states:
            bad_cell = model.state2cell(bad_state)
            current_cell = model.state2cell(current_state)
            
            # Manhattan distance
            dist = abs(bad_cell.row - current_cell.row) + abs(bad_cell.col - current_cell.col)
            
            if dist < min_dist_to_bad:
                min_dist_to_bad = dist
                closest_bad_state = bad_state
    
    # Check if path reached goal
    reached_goal = current_state == goal_state
    
    print(f"\n{name} Path Analysis:")
    print(f"  Path length: {step_count} steps")
    print(f"  Reached goal: {reached_goal}")
    print(f"  Minimum distance to cliff: {min_dist_to_bad}")
    
    # Convert path to cells for visualization
    path_cells = [model.state2cell(s) for s in path]
    print(f"  Path: {path_cells}")
    
    return {
        'path': path,
        'path_cells': path_cells,
        'path_length': step_count,
        'reached_goal': reached_goal,
        'min_dist_to_bad': min_dist_to_bad,
        'closest_bad_state': closest_bad_state
    }


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("===== Comparing SARSA and Q-learning on Cliff World =====")
    
    # Run comparison
    results = run_cliff_world_comparison(
        num_episodes=500,
        max_steps=200,
        alpha=0.1,
        epsilon=0.1,
        epsilon_decay=0.001,
        smooth_window=10,
        verbose=True,
        plot_save_name="cliff_world_comparison"
    )
    
    # Create model for path analysis
    model = Model(cliff_world)
    bad_states = [model.cell2state(cell) for cell in cliff_world.bad_cells]
    
    # Analyze paths
    sarsa_path = analyze_path_safety(
        model, 
        results['SARSA']['policy'], 
        model.start_state, 
        model.goal_state, 
        bad_states,
        name="SARSA"
    )
    
    q_learning_path = analyze_path_safety(
        model, 
        results['Q-learning']['policy'], 
        model.start_state, 
        model.goal_state, 
        bad_states,
        name="Q-learning"
    )
    
    # Compare path safety
    if sarsa_path['min_dist_to_bad'] > q_learning_path['min_dist_to_bad']:
        safer = "SARSA"
        dist_diff = sarsa_path['min_dist_to_bad'] - q_learning_path['min_dist_to_bad']
    elif q_learning_path['min_dist_to_bad'] > sarsa_path['min_dist_to_bad']:
        safer = "Q-learning"
        dist_diff = q_learning_path['min_dist_to_bad'] - sarsa_path['min_dist_to_bad']
    else:
        safer = "Both algorithms"
        dist_diff = 0
    
    print(f"\nSafety Comparison:")
    print(f"  {safer} maintains a safer distance from the cliff")
    if dist_diff > 0:
        print(f"  Difference in minimum cliff distance: {dist_diff}")
    
    print("\n===== Conclusion =====")
    print("This experiment demonstrates a key difference between SARSA and Q-learning:")
    print(" - SARSA (on-policy) learns a policy that accounts for exploration,")
    print("   leading to a safer path that avoids the cliff edge.")
    print(" - Q-learning (off-policy) learns the optimal deterministic policy,")
    print("   which often takes a riskier path closer to the cliff.")
    print("")
    print("In environments with large penalties and exploration, SARSA may perform")
    print("better during training because it accounts for the exploration policy.")
    print("However, if we can reduce exploration after training, Q-learning's policy")
    print("may yield higher returns due to finding the optimal deterministic path.")
    
    # Show plots
    plt.show() 