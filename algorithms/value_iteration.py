from typing import Callable
import numpy as np
from tqdm import tqdm

from model import Model, Actions


def synchronous_value_iteration(model: Model, max_iterations: int = 100, epsilon: float = 1e-6, 
                               iteration_monitor=None, bellman_update_monitor=None):
    """
    Synchronous Value Iteration algorithm for finding optimal value function and policy.
    
    Args:
        model: The model containing transitions and rewards
        max_iterations: Maximum number of iterations to run
        epsilon: Convergence threshold for value function updates
        iteration_monitor: Optional function to track iterations
        bellman_update_monitor: Optional function to track Bellman updates
        
    Returns:
        V: Optimal value function array
        pi: Optimal policy array
    """
    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,))
    
    for i in tqdm(range(max_iterations)):
        delta = 0  # Track maximum value change for convergence check
        
        # Track iteration if monitor is provided
        if iteration_monitor is not None:
            iteration_monitor(i)
        
        # Create a copy of the value function to update all states synchronously
        V_new = np.copy(V)
        
        for s in model.states:
            # Track Bellman update if monitor is provided
            if bellman_update_monitor is not None:
                bellman_update_monitor()
                
            # Calculate the value for each action and take the max
            action_values = [
                np.sum([
                    model.transition_probability(s, s_, a) * 
                    (model.reward(s, a) + model.gamma * V[s_])
                    for s_ in model.states
                ]) for a in Actions
            ]
            
            if len(action_values) > 0:
                best_value = np.max(action_values)
                best_action = np.argmax(action_values)
                
                # Update the value function and policy
                V_new[s] = best_value
                pi[s] = Actions(best_action)
                
                # Track the maximum change in value
                delta = max(delta, abs(V_new[s] - V[s]))
        
        # Update the value function after evaluating all states
        V = V_new
        
        # Check for convergence
        if delta < epsilon:
            print(f"✓ Synchronous Value Iteration converged after {i+1} iterations")
            print(f"  Final max-norm: δ = {delta:.8f} < ε = {epsilon:.8f}")
            break
        
        if i == max_iterations - 1:
            print(f"! Synchronous Value Iteration reached maximum iterations ({max_iterations})")
            print(f"  Final max-norm: δ = {delta:.8f}")
    
    return V, pi


def asynchronous_value_iteration(model: Model, max_iterations: int = 100, epsilon: float = 1e-6,
                                iteration_monitor=None, bellman_update_monitor=None):
    """
    Asynchronous Value Iteration algorithm for finding optimal value function and policy.
    Updates the value function in-place as soon as a state is evaluated.
    
    Args:
        model: The model containing transitions and rewards
        max_iterations: Maximum number of iterations to run
        epsilon: Convergence threshold for value function updates
        iteration_monitor: Optional function to track iterations
        bellman_update_monitor: Optional function to track Bellman updates
        
    Returns:
        V: Optimal value function array
        pi: Optimal policy array
    """
    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,))
    
    for i in tqdm(range(max_iterations)):
        delta = 0  # Track maximum value change for convergence check
        
        # Track iteration if monitor is provided
        if iteration_monitor is not None:
            iteration_monitor(i)
        
        for s in model.states:
            # Track Bellman update if monitor is provided
            if bellman_update_monitor is not None:
                bellman_update_monitor()
                
            # Store old value for convergence check
            v_old = V[s]
            
            # Calculate the value for each action and take the max
            action_values = [
                np.sum([
                    model.transition_probability(s, s_, a) * 
                    (model.reward(s, a) + model.gamma * V[s_])
                    for s_ in model.states
                ]) for a in Actions
            ]
            
            if len(action_values) > 0:
                best_value = np.max(action_values)
                best_action = np.argmax(action_values)
                
                # Update the value function and policy immediately
                V[s] = best_value
                pi[s] = Actions(best_action)
                
                # Track the maximum change in value
                delta = max(delta, abs(V[s] - v_old))
        
        # Check for convergence
        if delta < epsilon:
            print(f"✓ Asynchronous Value Iteration converged after {i+1} iterations")
            print(f"  Final max-norm: δ = {delta:.8f} < ε = {epsilon:.8f}")
            break
        
        if i == max_iterations - 1:
            print(f"! Asynchronous Value Iteration reached maximum iterations ({max_iterations})")
            print(f"  Final max-norm: δ = {delta:.8f}")
    
    return V, pi


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from world_config import cliff_world, small_world, grid_world
    from plot_vp import plot_vp

    # Create model using one of the provided world configurations
    model = Model(small_world)  # Try small_world to see policy differences
    
    # Fix visualization of -inf values
    def prepare_value_func_for_plot(V):
        V_copy = V.copy()
        V_copy[np.isneginf(V_copy)] = -1000  # Replace -inf with large negative number
        return V_copy
    
    # Run both synchronous and asynchronous value iteration
    print("Running synchronous value iteration...")
    V_sync, pi_sync = synchronous_value_iteration(model)
    
    print("\nRunning asynchronous value iteration...")
    V_async, pi_async = asynchronous_value_iteration(model)
    
    # Compare results
    print("\n===== Value Iteration Comparison =====")
    print("Value function difference (L2 norm):", np.linalg.norm(V_sync - V_async))
    policy_diff = np.sum(pi_sync != pi_async)
    print(f"Policy difference: {policy_diff} states ({policy_diff/len(model.states)*100:.2f}%)")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Synchronous Value Iteration")
    V_sync_plot = prepare_value_func_for_plot(V_sync)
    plot_vp(model, V_sync_plot, pi_sync)
    
    plt.subplot(1, 2, 2)
    plt.title("Asynchronous Value Iteration")
    V_async_plot = prepare_value_func_for_plot(V_async)
    plot_vp(model, V_async_plot, pi_async)
    
    plt.tight_layout()
    plt.show()
    
    # Now compare with policy iteration
    from policy_iteration import policy_iteration
    
    print("\nRunning policy iteration...")
    V_pi, pi_pi = policy_iteration(model)
    
    # Compare value iteration with policy iteration
    print("\n===== Comparison with Policy Iteration =====")
    print("Value function difference (Sync VI vs PI):", np.linalg.norm(V_sync - V_pi))
    policy_diff_sync = np.sum(pi_sync != pi_pi)
    print(f"Policy difference (Sync VI vs PI): {policy_diff_sync} states ({policy_diff_sync/len(model.states)*100:.2f}%)")
    
    print("Value function difference (Async VI vs PI):", np.linalg.norm(V_async - V_pi))
    policy_diff_async = np.sum(pi_async != pi_pi)
    print(f"Policy difference (Async VI vs PI): {policy_diff_async} states ({policy_diff_async/len(model.states)*100:.2f}%)")
    
    # Display specific states where policies differ
    if policy_diff_sync > 0 or policy_diff_async > 0:
        print("\nStates where policies differ:")
        count = 0
        for s in model.states:
            if pi_sync[s] != pi_pi[s] or pi_async[s] != pi_pi[s]:
                cell = model.state2cell(s)
                print(f"State {s} (Cell {cell}):")
                print(f"  PI action: {pi_pi[s]}")
                print(f"  Sync VI action: {pi_sync[s]}")
                print(f"  Async VI action: {pi_async[s]}")
                count += 1
                if count >= 5:  # Limit to 5 examples
                    print("... and more")
                    break 