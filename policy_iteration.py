from typing import Callable
from tqdm import tqdm
import numpy as np

from model import Model, Actions


def policy_iteration(model: Model, maxit: int = 100, iteration_monitor=None, bellman_update_monitor=None):
    """
    Policy Iteration algorithm for finding optimal value function and policy.
    
    Args:
        model: The model containing transitions and rewards
        maxit: Maximum number of iterations to run
        iteration_monitor: Optional function to track iterations
        bellman_update_monitor: Optional function to track Bellman updates
        
    Returns:
        V: Optimal value function array
        pi: Optimal policy array
    """
    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,))

    def compute_value(s, a, reward: Callable):
        return np.sum(
            [
                model.transition_probability(s, s_, a)
                * (reward(s, a) + model.gamma * V[s_])
                for s_ in model.states
            ]
        )

    def policy_evaluation():
        for s in model.states:
            # Track Bellman update if monitor is provided
            if bellman_update_monitor is not None:
                bellman_update_monitor()
                
            R = model.reward(s, pi[s])
            V[s] = compute_value(s, pi[s], lambda *_: R)

    def policy_improvement():
        policy_stable = True
        for s in model.states:
            old_action = pi[s]
            action_index = np.argmax(
                [compute_value(s, a, model.reward) for a in Actions]
            )
            pi[s] = Actions(action_index)
            if old_action != pi[s]:
                policy_stable = False
        return policy_stable

    for i in tqdm(range(maxit)):
        # Track iteration if monitor is provided
        if iteration_monitor is not None:
            iteration_monitor(i)
            
        # Perform policy evaluation step (multiple iterations for better convergence)
        for _ in range(5):
            policy_evaluation()
            
        # Perform policy improvement step    
        policy_stable = policy_improvement()
        
        if policy_stable:
            print(f"✓ Policy Iteration converged after {i+1} iterations")
            print(f"  Policy is stable (no changes after improvement)")
            break
            
        if i == maxit - 1:
            print(f"! Policy Iteration reached maximum iterations ({maxit})")
            print(f"  Policy may not have converged")

    return V, pi


def prepare_value_func_for_plot(V):
    """Prepare value function for plotting by replacing -inf with a large negative number."""
    V_copy = V.copy()
    # Replace -inf with a large negative number for visualization
    V_copy[np.isneginf(V_copy)] = -1000
    return V_copy


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from world_config import cliff_world, small_world, grid_world
    from plot_vp import plot_vp

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
    from value_iteration import synchronous_value_iteration, asynchronous_value_iteration
    
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
