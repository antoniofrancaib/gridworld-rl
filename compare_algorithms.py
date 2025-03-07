import time
import numpy as np
import matplotlib.pyplot as plt
# Make tabulate optional with clean fallback
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

from model import Model, Actions
from world_config import small_world, grid_world, cliff_world
from policy_iteration import policy_iteration
from value_iteration import synchronous_value_iteration, asynchronous_value_iteration


def count_operations(model, algorithm, **kwargs):
    """
    Function to count basic operations in each algorithm.
    Returns the number of iterations and operations performed.
    """
    operations = {
        'iterations': 0,
        'bellman_updates': 0,
        'transitions_computed': 0,
        'policy_evaluations': 0,
        'max_operations': 0
    }
    
    # Create a monitoring function to count iterations
    def monitor_iterations():
        operations['iterations'] += 1
        return operations['iterations']
    
    # Create a monitoring function to count Bellman updates
    def monitor_bellman_updates():
        operations['bellman_updates'] += 1
        return operations['bellman_updates']
    
    # Store the algorithm's iteration counter if needed
    if 'max_iterations' in kwargs:
        orig_max_it = kwargs['max_iterations']
        def iteration_monitor(i):
            operations['iterations'] = i + 1  # +1 because iterations are 0-indexed
            return i < orig_max_it
        kwargs['iteration_monitor'] = iteration_monitor
    
    # Store the algorithm's bellman update counter if needed
    if algorithm.__name__ in ['synchronous_value_iteration', 'asynchronous_value_iteration']:
        def bellman_update_monitor():
            operations['bellman_updates'] += 1
        kwargs['bellman_update_monitor'] = bellman_update_monitor
    
    # Run algorithm
    start_time = time.time()
    V, pi = algorithm(model, **kwargs)
    execution_time = time.time() - start_time
    
    # Calculate total Bellman updates if not already tracked
    if operations['bellman_updates'] == 0:
        if algorithm.__name__ == 'policy_iteration':
            # For policy iteration, count policy evaluations
            operations['policy_evaluations'] = operations['iterations'] * 5  # 5 evaluations per iteration
            operations['bellman_updates'] = operations['policy_evaluations'] * len(model.states)
        else:
            # For value iteration, each state update is a Bellman update
            operations['bellman_updates'] = operations['iterations'] * len(model.states)
    
    return V, pi, operations, execution_time


def analyze_policy_differences(model, pi1, pi2, name1, name2):
    """
    Analyze where two policies differ and provide possible explanations.
    
    Args:
        model: The environment model
        pi1, pi2: The two policies to compare
        name1, name2: Names of the algorithms for the policies
        
    Returns:
        A dictionary with analysis information
    """
    different_states = []
    
    for s in model.states:
        if s == model.fictional_end_state:
            continue  # Skip the fictional end state
            
        if pi1[s] != pi2[s]:
            cell = model.state2cell(s)
            different_states.append({
                'state': s,
                'cell': cell,
                f'{name1}_action': pi1[s],
                f'{name2}_action': pi2[s]
            })
    
    # Calculate value differences to analyze if states are near-optimal
    return {
        'different_states': different_states,
        'count': len(different_states),
        'percentage': len(different_states) / (len(model.states) - 1) * 100  # Exclude fictional end state
    }


def calculate_theoretical_complexity(model):
    """
    Calculate theoretical complexity of algorithms based on state and action space.
    
    Returns:
        Dictionary with theoretical complexity estimates
    """
    num_states = len(model.states)
    num_actions = len(Actions)
    
    return {
        'Policy Iteration': {
            'policy_evaluation': f"O({num_states}²) per evaluation",
            'policy_improvement': f"O({num_states}²·{num_actions})",
            'total': f"O({num_states}³ + {num_states}²·{num_actions}·iterations)"
        },
        'Synchronous VI': {
            'complexity': f"O({num_states}²·{num_actions}·iterations)"
        },
        'Asynchronous VI': {
            'complexity': f"O({num_states}²·{num_actions}·iterations)",
            'note': "Typically faster convergence in practice compared to synchronous updates"
        }
    }


def prepare_value_func_for_plot(V):
    """Prepare value function for plotting by replacing -inf with a large negative number."""
    V_copy = V.copy()
    # Replace -inf with a large negative number for visualization
    V_copy[np.isneginf(V_copy)] = -1000
    return V_copy


def explain_policy_differences(model, policies, names, world_name):
    """
    Provide a detailed explanation of policy differences.
    
    Args:
        model: The environment model
        policies: Dictionary of policies from different algorithms
        names: List of algorithm names
        world_name: Name of the world being analyzed
    """
    differences = {}
    
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            diff = analyze_policy_differences(model, policies[name1], policies[name2], name1, name2)
            differences[f"{name1} vs {name2}"] = diff
    
    print("\n===== Policy Difference Analysis =====")
    
    if all(diff['count'] == 0 for diff in differences.values()):
        print(f"✓ All policies for {world_name} are identical.")
        return
    
    # Summarize differences in a table
    print(f"Policy differences in {world_name}:")
    
    table_data = []
    for comparison, diff in differences.items():
        table_data.append([
            comparison,
            diff['count'],
            f"{diff['percentage']:.2f}%"
        ])
    
    headers = ["Comparison", "Different States", "Percentage"]
    
    if tabulate:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print(f"{headers[0]:<20} {headers[1]:<20} {headers[2]:<15}")
        print("-" * 55)
        for row in table_data:
            print(f"{row[0]:<20} {row[1]:<20} {row[2]:<15}")
    
    # Show examples of differences
    for comparison, diff in differences.items():
        if diff['count'] > 0:
            print(f"\nExample differences for {comparison}:")
            for i, state_diff in enumerate(diff['different_states'][:3]):  # Show up to 3 examples
                state = state_diff['state']
                cell = state_diff['cell']
                action1 = state_diff[f"{names[0]}_action"] if f"{names[0]}_action" in state_diff else state_diff[f"{comparison.split(' vs ')[0]}_action"]
                action2 = state_diff[f"{names[1]}_action"] if f"{names[1]}_action" in state_diff else state_diff[f"{comparison.split(' vs ')[1]}_action"]
                
                # Determine if the cell is near a special location
                near_special = ""
                if model.state2cell(model.start_state) == cell:
                    near_special = "(START CELL)"
                elif model.state2cell(model.goal_state) == cell:
                    near_special = "(GOAL CELL)"
                elif cell in model.world.bad_cells:
                    near_special = "(BAD CELL)"
                
                print(f"  State {state} {near_special} at Cell{cell}:")
                print(f"    → {comparison.split(' vs ')[0]} action: {action1}")
                print(f"    → {comparison.split(' vs ')[1]} action: {action2}")
    
    # Provide explanation for differences
    print("\nExplanation for Policy Differences:")
    
    if world_name == "Small World":
        print("  In Small World, differences likely arise due to:")
        print("   1. Multiple optimal policies may exist due to the small state space")
        print("   2. Stochastic transitions (prob_good_trans = 0.8, bias = 0.5) create")
        print("      multiple paths with similar expected returns")
        print("   3. The discount factor (γ = 0.9) reduces the impact of future rewards")
        print("   4. Different convergence paths in the algorithms can lead to different")
        print("      but equally optimal policies")
        print("   5. Numerical precision differences between algorithms")
    elif world_name == "Grid World":
        print("  In Grid World, differences may be due to:")
        print("   1. The larger state space creates more potential for convergence to")
        print("      different local optima, especially with stochastic transitions")
        print("   2. Different update orders between algorithms (especially async vs sync)")
        print("   3. The presence of obstacles creates multiple possible paths with")
        print("      similar values")
    elif world_name == "Cliff World":
        print("  In Cliff World, differences usually relate to:")
        print("   1. Risk-avoidance vs optimal-path tradeoffs near the cliff")
        print("   2. The return-to-start penalty creates multiple strategies with")
        print("      similar expected returns")
    
    # Theoretically, both algorithms should converge to the same optimal policy
    print("\nTheoretical Note: Both Value Iteration and Policy Iteration should converge")
    print("to the same optimal policy in the limit. Any differences are typically due to:")
    print(" - Different convergence thresholds")
    print(" - Finite iteration counts")
    print(" - Numerical precision issues")
    print(" - Existence of multiple optimal policies")


def run_comparison(world_name, world_config, max_iterations=100, epsilon=1e-6):
    """Run comparison between policy iteration and both value iteration variants"""
    model = Model(world_config)
    
    results = {}
    
    # Run policy iteration
    print(f"\nRunning Policy Iteration on {world_name}...")
    V_pi, pi_pi, ops_pi, time_pi = count_operations(model, policy_iteration, maxit=max_iterations)
    results['Policy Iteration'] = {
        'value_function': V_pi,
        'policy': pi_pi,
        'operations': ops_pi,
        'time': time_pi
    }
    
    # Run synchronous value iteration
    print(f"Running Synchronous Value Iteration on {world_name}...")
    V_sync, pi_sync, ops_sync, time_sync = count_operations(
        model, synchronous_value_iteration, max_iterations=max_iterations, epsilon=epsilon
    )
    results['Synchronous VI'] = {
        'value_function': V_sync,
        'policy': pi_sync,
        'operations': ops_sync,
        'time': time_sync
    }
    
    # Run asynchronous value iteration
    print(f"Running Asynchronous Value Iteration on {world_name}...")
    V_async, pi_async, ops_async, time_async = count_operations(
        model, asynchronous_value_iteration, max_iterations=max_iterations, epsilon=epsilon
    )
    results['Asynchronous VI'] = {
        'value_function': V_async,
        'policy': pi_async,
        'operations': ops_async,
        'time': time_async
    }
    
    # Print comparison results
    print(f"\n--- {world_name} Comparison Results ---")
    
    # Policy match checking
    policies = {
        'Policy Iteration': pi_pi,
        'Synchronous VI': pi_sync,
        'Asynchronous VI': pi_async
    }
    alg_names = list(policies.keys())
    
    # Compare iterations and operations
    print("\nPerformance Metrics:")
    headers = ["Algorithm", "Iterations", "Bellman Updates", "Time (s)"]
    table_data = []
    
    for name, data in results.items():
        table_data.append([
            name, 
            data['operations']['iterations'],
            data['operations']['bellman_updates'],
            f"{data['time']:.4f}"
        ])
    
    if tabulate:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print(f"{headers[0]:<20} {headers[1]:<12} {headers[2]:<16} {headers[3]:<10}")
        print("-" * 60)
        for row in table_data:
            print(f"{row[0]:<20} {row[1]:<12} {row[2]:<16} {row[3]:<10}")
    
    # Calculate and display theoretical complexity
    complexity = calculate_theoretical_complexity(model)
    print("\nTheoretical Complexity Estimates:")
    for alg, info in complexity.items():
        print(f"  - {alg}:")
        for metric, value in info.items():
            print(f"    {metric}: {value}")
    
    # Analyze policy differences
    explain_policy_differences(model, policies, alg_names, world_name)
    
    return results


def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart."""
    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space = -space
            # Vertically align label at top
            va = 'top'

        # Create annotation
        ax.annotate(
            f'{y_value:.1f}',               # Use formatted value
            (x_value, y_value),             # Place label at end of the bar
            xytext=(0, space),              # Vertically shift label by `space`
            textcoords="offset points",     # Interpret `xytext` as offset in points
            ha='center',                    # Horizontally center label
            va=va,                          # Vertically align as previously determined
            fontsize=8)                     # Small font size


if __name__ == "__main__":
    # Set consistent figure style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Run comparison on different world configurations
    worlds = {
        "Small World": small_world,
        "Grid World": grid_world,
        "Cliff World": cliff_world
    }
    
    all_results = {}
    
    for name, world in worlds.items():
        all_results[name] = run_comparison(name, world)
    
    # Create visualization for each world
    for world_name, world_config in worlds.items():
        model = Model(world_config)
        
        # Create multi-panel figure
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(f"Reinforcement Learning Algorithm Comparison: {world_name}", fontsize=18)
        
        # Set up the grid layout for the figure
        gs = plt.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.3)
        
        # Fix for visualization: prepare value functions for plotting
        plot_vp = __import__('plot_vp').plot_vp
        
        # Plot policy iteration
        ax1 = fig.add_subplot(gs[0, 0])
        V_pi_plot = prepare_value_func_for_plot(all_results[world_name]["Policy Iteration"]["value_function"])
        plot_vp(model, V_pi_plot, all_results[world_name]["Policy Iteration"]["policy"])
        ax1.set_title("Policy Iteration", fontsize=14)
        
        # Plot synchronous VI
        ax2 = fig.add_subplot(gs[0, 1])
        V_sync_plot = prepare_value_func_for_plot(all_results[world_name]["Synchronous VI"]["value_function"])
        plot_vp(model, V_sync_plot, all_results[world_name]["Synchronous VI"]["policy"])
        ax2.set_title("Synchronous Value Iteration", fontsize=14)
        
        # Plot asynchronous VI
        ax3 = fig.add_subplot(gs[0, 2])
        V_async_plot = prepare_value_func_for_plot(all_results[world_name]["Asynchronous VI"]["value_function"])
        plot_vp(model, V_async_plot, all_results[world_name]["Asynchronous VI"]["policy"])
        ax3.set_title("Asynchronous Value Iteration", fontsize=14)
        
        # Plot iterations comparison
        ax4 = fig.add_subplot(gs[1, 0])
        alg_names = list(all_results[world_name].keys())
        
        x_pos = np.arange(len(alg_names))
        iterations = [all_results[world_name][alg]['operations']['iterations'] for alg in alg_names]
        bars = ax4.bar(x_pos, iterations, width=0.5, color='#5DA5DA', edgecolor='black')
        add_value_labels(ax4)
        
        ax4.set_ylabel('Iterations', fontsize=12)
        ax4.set_title('Iterations to Convergence', fontsize=14)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(alg_names, rotation=30, ha='right')
        
        # Plot Bellman updates comparison
        ax5 = fig.add_subplot(gs[1, 1])
        updates = [all_results[world_name][alg]['operations']['bellman_updates'] for alg in alg_names]
        bars = ax5.bar(x_pos, updates, width=0.5, color='#F17CB0', edgecolor='black')
        add_value_labels(ax5)
        
        ax5.set_ylabel('Bellman Updates', fontsize=12)
        ax5.set_title('Total Bellman Updates', fontsize=14)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(alg_names, rotation=30, ha='right')
        
        # Plot execution time comparison
        ax6 = fig.add_subplot(gs[1, 2])
        times = [all_results[world_name][alg]['time'] for alg in alg_names]
        bars = ax6.bar(x_pos, times, width=0.5, color='#60BD68', edgecolor='black')
        add_value_labels(ax6)
        
        ax6.set_ylabel('Time (seconds)', fontsize=12)
        ax6.set_title('Execution Time', fontsize=14)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(alg_names, rotation=30, ha='right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for main title
        plt.savefig(f'{world_name.replace(" ", "_").lower()}_comparison.png', dpi=300, bbox_inches='tight')
    
    # Create a final summary figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Algorithm Performance Across Environments', fontsize=18)
    
    # Set up data for comparison across worlds
    world_names = list(worlds.keys())
    alg_names = list(all_results[world_names[0]].keys())
    x = np.arange(len(world_names))
    width = 0.25
    
    # Plot iterations comparison
    for i, alg in enumerate(alg_names):
        iterations = [all_results[world][alg]['operations']['iterations'] for world in world_names]
        bars = axes[0].bar(x + (i-1)*width, iterations, width, label=alg)
    
    axes[0].set_ylabel('Iterations', fontsize=12)
    axes[0].set_title('Iterations to Convergence', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(world_names, rotation=0)
    axes[0].legend(loc='upper left')
    
    # Plot Bellman updates comparison
    for i, alg in enumerate(alg_names):
        updates = [all_results[world][alg]['operations']['bellman_updates'] for world in world_names]
        bars = axes[1].bar(x + (i-1)*width, updates, width, label=alg)
    
    axes[1].set_ylabel('Bellman Updates', fontsize=12)
    axes[1].set_title('Total Bellman Updates', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(world_names, rotation=0)
    axes[1].legend(loc='upper left')
    
    # Plot execution time comparison
    for i, alg in enumerate(alg_names):
        times = [all_results[world][alg]['time'] for world in world_names]
        bars = axes[2].bar(x + (i-1)*width, times, width, label=alg)
    
    axes[2].set_ylabel('Time (seconds)', fontsize=12)
    axes[2].set_title('Execution Time', fontsize=14)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(world_names, rotation=0)
    axes[2].legend(loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for main title
    plt.savefig('overall_comparison.png', dpi=300, bbox_inches='tight')
    
    # Show all plots
    plt.show()

    # Print final summary
    print("\n===== SUMMARY OF RESULTS =====")
    print("1. Stopping Criteria for Value Iteration:")
    print("   - Both Sync and Async VI use max|V(s) - V'(s)| < ε = 1e-6 as convergence criteria")
    print("   - This ensures the value function has stabilized to near-optimal values")
    print("   - Provides a principled way to terminate the algorithm with theoretical guarantees")
    
    print("\n2. Synchronous vs Asynchronous Value Iteration:")
    for world in world_names:
        sync_iter = all_results[world]['Synchronous VI']['operations']['iterations']
        async_iter = all_results[world]['Asynchronous VI']['operations']['iterations']
        sync_time = all_results[world]['Synchronous VI']['time']
        async_time = all_results[world]['Asynchronous VI']['time']
        
        print(f"   {world}:")
        print(f"     - Sync VI: {sync_iter} iterations, {sync_time:.4f}s")
        print(f"     - Async VI: {async_iter} iterations, {async_time:.4f}s")
        print(f"     - {'Async' if async_iter < sync_iter else 'Sync'} VI required fewer iterations")
        print(f"     - {'Async' if async_time < sync_time else 'Sync'} VI was faster")
    
    print("\n3. Policy Iteration vs Value Iteration:")
    for world in world_names:
        pi_iter = all_results[world]['Policy Iteration']['operations']['iterations']
        pi_time = all_results[world]['Policy Iteration']['time']
        vi_time = all_results[world]['Synchronous VI']['time']
        
        print(f"   {world}:")
        print(f"     - PI required {pi_iter} iterations")
        print(f"     - PI was {'faster' if pi_time < vi_time else 'slower'} than Sync VI")
        
    print("\n4. Computational Complexity Analysis:")
    print("   - Policy Iteration: O(|S|³ + |S|²|A|·iterations)")
    print("     * Requires fewer iterations but each iteration is more expensive")
    print("   - Value Iteration: O(|S|²|A|·iterations)")
    print("     * Requires more iterations but each iteration is less expensive")
    print("   - Asynchronous VI is generally more efficient in practice as it")
    print("     incorporates new information immediately") 