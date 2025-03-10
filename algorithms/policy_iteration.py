from typing import Callable
from tqdm import tqdm
import numpy as np

import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.model import Model, Actions


def policy_iteration(model, maxit=100, delta=1e-10, gamma=None, iteration_monitor=None, policy_evaluation_monitor=None):
    """
    Implementation of policy iteration algorithm that exactly matches the model's transitions and rewards.
    
    Args:
        model: The environment model
        maxit: Maximum number of iterations
        delta: Convergence threshold for policy evaluation
        gamma: Discount factor (if None, uses model.world.gamma)
        iteration_monitor: Optional callback after each iteration
        policy_evaluation_monitor: Optional callback during policy evaluation
        
    Returns:
        V: The value function
        pi: The policy
    """
    if gamma is None:
        gamma = model.world.gamma
    
    # Initialize with a random policy
    pi = {}
    for s in model.states:
        pi[s] = np.random.choice(list(Actions))
    
    # Set policy for special states
    if model.goal_state in model.states:
        pi[model.goal_state] = Actions.UP  # Arbitrary action for goal state
    if model.fictional_end_state in model.states:
        pi[model.fictional_end_state] = Actions.UP  # Arbitrary action for fictional end state
    for s in model.obstacle_states:
        if s in model.states:
            pi[s] = Actions.UP  # Arbitrary action for obstacle states
    
    # Initialize value function
    V = np.zeros(len(model.states))
    
    # Set initial values for special states
    V[model.fictional_end_state] = 0.0
    if model.goal_state < len(V):
        # Goal state value is reward_goal / (1-gamma) in the infinite horizon case
        V[model.goal_state] = model.world.reward_goal
    
    # Policy iteration
    for it in range(maxit):
        # Call iteration monitor if provided
        if iteration_monitor is not None:
            if not iteration_monitor(it):
                break
                
        # Policy evaluation (to convergence with high iteration limit)
        V = policy_evaluation(
            model, pi, V, gamma=gamma, delta=delta, maxit=5000, 
            evaluation_monitor=policy_evaluation_monitor
        )
        
        # Policy improvement
        policy_stable = True
        for s in model.states:
            # Skip special states
            if s == model.fictional_end_state or s == model.goal_state or s in model.obstacle_states:
                continue
            
            old_action = pi[s]
            
            # Compute the expected value for each action
            best_action = None
            best_value = -np.inf
            
            for a in Actions:
                # Calculate value using model's transition probabilities
                value = 0.0
                for next_state in model.states:
                    prob = model.transition_probability(s, next_state, a)
                    if prob > 0:
                        reward = model.reward(s, a)
                        value += prob * (reward + gamma * V[next_state])
                
                if value > best_value:
                    best_value = value
                    best_action = a
            
            pi[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        # If policy is stable, we've converged
        if policy_stable:
            if iteration_monitor is not None:
                iteration_monitor(it+1)  # Final update for monitoring
            print(f"✓ Policy Iteration converged after {it+1} iterations")
            break
            
        if it == maxit - 1:
            print(f"! Policy Iteration reached maximum iterations ({maxit})")
    
    return V, pi


def compute_expected_value(model, state, action, V, gamma):
    """
    Compute the expected value of taking action 'a' in state 's' under current value function V.
    This function uses the exact same transition and reward logic as the Model class.
    
    Args:
        model: The environment model
        state: The current state
        action: The action to evaluate
        V: The current value function
        gamma: Discount factor
        
    Returns:
        The expected value
    """
    # Special case handling for terminal states
    if state == model.fictional_end_state:
        return 0.0  # Terminal state has value 0
    
    if state == model.goal_state or state in model.obstacle_states:
        # Goal and obstacle states transition to fictional end state with reward
        reward = model.reward(state, action)
        return reward + gamma * V[model.fictional_end_state]
        
    # For bad states in cliff world, special handling
    if state in model.bad_states and model.world.return_to_start_from_bad_state:
        reward = model.reward(state, action)
        return reward + gamma * V[model.start_state]
    
    # For normal states, calculate expected value using model's transition probabilities
    expected_value = 0.0
    
    # Get all possible next states and their probabilities using the model's method
    for next_state in model.states:
        # Get transition probability using the model's method
        prob = model.transition_probability(state, next_state, action)
        
        if prob > 0:
            # Get reward using the model's method
            reward = model.reward(state, action)
            expected_value += prob * (reward + gamma * V[next_state])
    
    return expected_value


def policy_evaluation(model, pi, V, gamma=0.9, delta=1e-6, maxit=100, evaluation_monitor=None):
    """
    Implementation of iterative policy evaluation.
    
    Args:
        model: Model object containing dynamics and rewards
        pi: Policy to evaluate
        V: Current value function estimate
        gamma: Discount factor (this parameter is overridden by the caller)
        delta: Convergence criterion
        maxit: Maximum number of iterations
        evaluation_monitor: Optional function to call each time policy evaluation is run
    
    Returns:
        V: Updated value function
    """
    V = V.copy()  # Make a copy to avoid modifying the input
    
    # Call the evaluation monitor if provided
    if evaluation_monitor is not None:
        evaluation_monitor()
    
    # Handle fictional end state specially - always has value 0
    V[model.fictional_end_state] = 0.0
    
    # Handle goal state specially - its value is reward_goal
    if model.goal_state < len(V):
        V[model.goal_state] = model.world.reward_goal
    
    # Evaluate policy until convergence or max iterations
    for _ in range(maxit):
        delta_value = 0
        
        for s in model.states:
            # Skip fictional end state, goal state, and obstacle states
            if s == model.fictional_end_state or s == model.goal_state or s in model.obstacle_states:
                continue
                
            # Store old value for convergence check
            v_old = V[s]
            
            # Look up the action from the policy
            a = pi[s]
            
            # Calculate value using exact same model transitions and rewards
            new_value = 0.0
            for next_state in model.states:
                trans_prob = model.transition_probability(s, next_state, a)
                if trans_prob > 0:
                    reward = model.reward(s, a)
                    new_value += trans_prob * (reward + gamma * V[next_state])
            
            # Update value function
            V[s] = new_value
            
            # Track maximum change for convergence check
            delta_value = max(delta_value, abs(new_value - v_old))
        
        # Check for convergence
        if delta_value < delta:
            break
    
    return V


def prepare_value_func_for_plot(V):
    """Prepare value function for plotting by replacing -inf with a large negative number."""
    V_copy = V.copy()
    # Replace -inf with a large negative number for visualization
    V_copy[np.isneginf(V_copy)] = -1000
    return V_copy


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from environment.world_config import cliff_world, small_world, grid_world
    from utils.plot_vp import plot_vp

    # Create model using one of the provided world configurations
    model = Model(small_world)
    
    print("Running Policy Iteration...")
    V, pi = policy_iteration(model)
    
    # Plot the result
    plt.figure(figsize=(8, 6))
    V_plot = prepare_value_func_for_plot(V)
    plot_vp(model, V_plot, pi)
    plt.title("Policy Iteration: Value Function & Policy")
    plt.tight_layout()
    plt.show()
    
    # Compare with Value Iteration
    print("\nComparing with Value Iteration...")
    from algorithms.value_iteration import synchronous_value_iteration, asynchronous_value_iteration
    
    print("\nRunning Synchronous Value Iteration...")
    V_sync, pi_sync = synchronous_value_iteration(model)
    
    print("\nRunning Asynchronous Value Iteration...")
    V_async, pi_async = asynchronous_value_iteration(model)
    
    # Check if policies match
    policy_match_sync = np.array_equal(pi, pi_sync)
    policy_match_async = np.array_equal(pi, pi_async)
    
    print("\n===== Policy Comparison =====")
    print(f"Policy match (PI vs Sync VI): {policy_match_sync}")
    print(f"Policy match (PI vs Async VI): {policy_match_async}")
    
    if not policy_match_sync or not policy_match_async:
        print("\nPolicy differences found. This might be due to:")
        print("1. Multiple optimal policies exist")
        print("2. Different convergence criteria")
        print("3. Stochastic transitions creating similar-value alternatives")
        print("4. Numerical precision differences")
