import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Make tabulate optional with clean fallback
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

# Import from utils package according to new directory structure
from utils.plot_vp import plot_vp
    
from environment.model import Model, Actions
from environment.world_config import small_world, grid_world, cliff_world
from algorithms.policy_iteration import policy_iteration
from algorithms.value_iteration import synchronous_value_iteration, asynchronous_value_iteration


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
    
    # Fix for policy iteration which uses 'maxit' instead of 'max_iterations'
    if 'maxit' in kwargs and algorithm.__name__ == 'policy_iteration':
        # Make a copy of the original max iterations
        orig_max_it = kwargs['maxit']
        
        # Create a wrapper for the iteration monitor
        def policy_iteration_monitor(i):
            # Update the iteration count - add 1 because i is 0-indexed
            operations['iterations'] = i + 1
            return True  # Always continue until policy_iteration's own convergence check
            
        # Add our monitor to kwargs
        kwargs['iteration_monitor'] = policy_iteration_monitor
        
    # Handle the standard max_iterations parameter for value iteration
    elif 'max_iterations' in kwargs:
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
    elif algorithm.__name__ == 'policy_iteration':
        def policy_update_monitor():
            operations['policy_evaluations'] += 1
        kwargs['policy_evaluation_monitor'] = policy_update_monitor
    
    # Run algorithm
    start_time = time.time()
    V, pi = algorithm(model, **kwargs)
    execution_time = time.time() - start_time
    
    # Calculate total Bellman updates if not already tracked
    if operations['bellman_updates'] == 0:
        if algorithm.__name__ == 'policy_iteration':
            # For policy iteration:
            # 1. Each iteration involves a full policy evaluation
            # 2. Each policy evaluation requires multiple sweeps through the state space 
            #    until convergence (typically 10-20 sweeps)
            # 3. Each sweep updates each state once
            if operations['policy_evaluations'] > 0:
                # If we tracked policy evaluations, use that
                # Assume ~15 sweeps per policy evaluation on average
                operations['bellman_updates'] = operations['policy_evaluations'] * len(model.states) * 15
            else:
                # Each iteration of policy iteration performs one policy evaluation
                # Estimate ~15 sweeps per evaluation
                operations['policy_evaluations'] = operations['iterations']
                operations['bellman_updates'] = operations['iterations'] * len(model.states) * 15
        else:
            # For both sync and async value iteration:
            # Each iteration involves exactly one update per state
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
    """
    Prepare value function for plotting by replacing -inf with a large negative number.
    Works with both NumPy arrays and dictionaries.
    
    Args:
        V: Value function (either NumPy array or dictionary mapping states to values)
        
    Returns:
        Processed copy of V suitable for visualization
    """
    # Make a copy to avoid modifying the original
    V_copy = V.copy()
    
    # Handle NumPy arrays
    if isinstance(V_copy, np.ndarray):
        # Replace -inf values with a large negative number
        V_copy[np.isneginf(V_copy)] = -1000
    # Handle dictionaries
    elif isinstance(V_copy, dict):
        for state, value in V_copy.items():
            if np.isneginf(value):
                V_copy[state] = -1000
    
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
    
    print(f"\n===== Policy Difference Analysis for {world_name} =====")
    
    if all(diff['count'] == 0 for diff in differences.values()):
        print(f"✓ All policies for {world_name} are identical - all algorithms converged to the same optimal policy.")
        return differences
    
    # Summarize differences in a table
    print(f"Policy differences in {world_name}:")
    
    table_data = []
    for comparison, diff in differences.items():
        table_data.append([
            comparison,
            diff['count'],
            f"{diff['percentage']:.2f}%"
        ])
    
    headers = ["Algorithm Comparison", "Different States", "Percentage of States"]
    
    if tabulate:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print(f"{headers[0]:<30} {headers[1]:<20} {headers[2]:<20}")
        print("-" * 72)
        for row in table_data:
            print(f"{row[0]:<30} {row[1]:<20} {row[2]:<20}")
    
    # Show examples of differences
    for comparison, diff in differences.items():
        if diff['count'] > 0:
            print(f"\nExample differences for {comparison}:")
            for i, state_diff in enumerate(diff['different_states'][:3]):  # Show up to 3 examples
                state = state_diff['state']
                cell = state_diff['cell']
                action1 = state_diff[f"{comparison.split(' vs ')[0]}_action"]
                action2 = state_diff[f"{comparison.split(' vs ')[1]}_action"]
                
                # Determine if the cell is near a special location
                near_special = ""
                if model.state2cell(model.start_state) == cell:
                    near_special = "(START CELL)"
                elif model.state2cell(model.goal_state) == cell:
                    near_special = "(GOAL CELL)"
                elif cell in model.world.bad_cells:
                    near_special = "(BAD CELL)"
                
                print(f"  State {state} {near_special} at Cell{cell}:")
                print(f"    → {comparison.split(' vs ')[0]} chooses: {action1}")
                print(f"    → {comparison.split(' vs ')[1]} chooses: {action2}")
    
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
        print("   3. Different algorithms may take different approaches to navigating")
        print("      the risk-reward tradeoff between shortest path and avoiding the cliff")
    
    # Theoretically, both algorithms should converge to the same optimal policy
    print("\nTheoretical Note: Both Value Iteration and Policy Iteration should converge")
    print("to the same optimal policy in the limit. Any differences are typically due to:")
    print(" - Different convergence thresholds")
    print(" - Finite iteration counts")
    print(" - Numerical precision issues")
    print(" - Existence of multiple optimal policies with identical values")
    
    return differences


def run_comparison(world_name, world_config, max_iterations=100, epsilon=1e-6):
    """
    Run comparison between policy iteration and both value iteration variants.
    
    Args:
        world_name: Name of the world configuration
        world_config: World configuration dictionary
        max_iterations: Maximum iterations for algorithms
        epsilon: Convergence threshold for value iteration
        
    Returns:
        Dictionary with results from all algorithms
    """
    model = Model(world_config)
    
    results = {}
    
    # Create a results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run policy iteration
    print(f"\nRunning Policy Iteration on {world_name}...")
    try:
        V_pi, pi_pi, ops_pi, time_pi = count_operations(model, policy_iteration, maxit=max_iterations)
        results['Policy Iteration'] = {
            'value_function': V_pi,
            'policy': pi_pi,
            'operations': ops_pi,
            'time': time_pi
        }
        print(f"  ✓ Completed in {ops_pi['iterations']} iterations and {time_pi:.4f} seconds")
    except Exception as e:
        print(f"  ✗ Error during Policy Iteration: {e}")
        return None
    
    # Run synchronous value iteration
    print(f"Running Synchronous Value Iteration on {world_name}...")
    try:
        V_sync, pi_sync, ops_sync, time_sync = count_operations(
            model, synchronous_value_iteration, max_iterations=max_iterations, epsilon=epsilon
        )
        results['Synchronous VI'] = {
            'value_function': V_sync,
            'policy': pi_sync,
            'operations': ops_sync,
            'time': time_sync
        }
        print(f"  ✓ Completed in {ops_sync['iterations']} iterations and {time_sync:.4f} seconds")
    except Exception as e:
        print(f"  ✗ Error during Synchronous Value Iteration: {e}")
        return None
    
    # Run asynchronous value iteration
    print(f"Running Asynchronous Value Iteration on {world_name}...")
    try:
        V_async, pi_async, ops_async, time_async = count_operations(
            model, asynchronous_value_iteration, max_iterations=max_iterations, epsilon=epsilon
        )
        results['Asynchronous VI'] = {
            'value_function': V_async,
            'policy': pi_async,
            'operations': ops_async,
            'time': time_async
        }
        print(f"  ✓ Completed in {ops_async['iterations']} iterations and {time_async:.4f} seconds")
    except Exception as e:
        print(f"  ✗ Error during Asynchronous Value Iteration: {e}")
        return None
    
    # Print comparison results
    print(f"\n===== {world_name} Comparison Results =====")
    
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
    policy_diff = explain_policy_differences(model, policies, alg_names, world_name)
    
    # Save summarized results
    with open(f"results/{world_name.replace(' ', '_').lower()}_summary.txt", 'w') as f:
        f.write(f"===== {world_name} Algorithm Comparison =====\n\n")
        f.write("Performance Metrics:\n")
        for name, data in results.items():
            f.write(f"{name}:\n")
            f.write(f"  - Iterations: {data['operations']['iterations']}\n")
            f.write(f"  - Bellman Updates: {data['operations']['bellman_updates']}\n")
            f.write(f"  - Execution Time: {data['time']:.4f}s\n\n")
        
        if policy_diff:
            f.write("Policy Differences:\n")
            for comparison, diff in policy_diff.items():
                f.write(f"{comparison}: {diff['count']} states differ ({diff['percentage']:.2f}%)\n")
    
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
        
        # Create multi-panel figure with explicit figure number to prevent extra plots
        plt.figure(figsize=(18, 10), clear=True)
        fig = plt.gcf()
        fig.suptitle(f"Reinforcement Learning Algorithm Comparison: {world_name}", fontsize=18)
        
        # Set up the grid layout for the figure
        gs = plt.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.4)
        
        # Clear any existing subplots to prevent extra plots
        plt.clf()
        
        # Plot policy iteration
        ax1 = fig.add_subplot(gs[0, 0])
        V_pi_plot = prepare_value_func_for_plot(all_results[world_name]["Policy Iteration"]["value_function"])
        plot_vp(model, V_pi_plot, all_results[world_name]["Policy Iteration"]["policy"], ax=ax1, title=f"Policy Iteration\n{world_name}")
        ax1.set_title("Policy Iteration\nValue Function & Policy", fontsize=14)
        # Add colorbar explanation
        # ax1.text(0.5, -0.15, "Colors: Value function (brighter = higher value)", transform=ax1.transAxes, 
        #        ha='center', fontsize=8, style='italic')
        
        # Plot synchronous VI
        ax2 = fig.add_subplot(gs[0, 1])
        V_sync_plot = prepare_value_func_for_plot(all_results[world_name]["Synchronous VI"]["value_function"])
        plot_vp(model, V_sync_plot, all_results[world_name]["Synchronous VI"]["policy"], ax=ax2, title=f"Synchronous VI\n{world_name}")
        ax2.set_title("Synchronous Value Iteration\nValue Function & Policy", fontsize=14)
        
        # Plot asynchronous VI
        ax3 = fig.add_subplot(gs[0, 2])
        V_async_plot = prepare_value_func_for_plot(all_results[world_name]["Asynchronous VI"]["value_function"])
        plot_vp(model, V_async_plot, all_results[world_name]["Asynchronous VI"]["policy"], ax=ax3, title=f"Asynchronous VI\n{world_name}")
        ax3.set_title("Asynchronous Value Iteration\nValue Function & Policy", fontsize=14)
        
        # Add legend for policy arrows
        # ax3.text(0.5, -0.15, "Arrows: Optimal action directions from policy", transform=ax3.transAxes, 
        #        ha='center', fontsize=8, style='italic')
        
        # Plot iterations comparison
        ax4 = fig.add_subplot(gs[1, 0])
        alg_names = list(all_results[world_name].keys())
        
        x_pos = np.arange(len(alg_names))
        iterations = [all_results[world_name][alg]['operations']['iterations'] for alg in alg_names]
        bars = ax4.bar(x_pos, iterations, width=0.5, color='#5DA5DA', edgecolor='black')
        add_value_labels(ax4)
        
        ax4.set_ylabel('Number of Iterations', fontsize=12)
        ax4.set_title(f'Iterations to Convergence\n({world_name})', fontsize=14)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(alg_names, rotation=30, ha='right')
        ax4.text(0.5, -0.25, "Lower is better - fewer iterations needed", transform=ax4.transAxes, 
                ha='center', fontsize=8, style='italic')
        
        # Plot Bellman updates comparison
        ax5 = fig.add_subplot(gs[1, 1])
        updates = [all_results[world_name][alg]['operations']['bellman_updates'] for alg in alg_names]
        bars = ax5.bar(x_pos, updates, width=0.5, color='#F17CB0', edgecolor='black')
        add_value_labels(ax5)
        
        ax5.set_ylabel('Number of Updates', fontsize=12)
        ax5.set_title(f'Total Bellman Updates\n({world_name})', fontsize=14)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(alg_names, rotation=30, ha='right')
        ax5.text(0.5, -0.25, "Lower is better - computational efficiency", transform=ax5.transAxes, 
                ha='center', fontsize=8, style='italic')
        
        # Plot execution time comparison
        ax6 = fig.add_subplot(gs[1, 2])
        times = [all_results[world_name][alg]['time'] for alg in alg_names]
        bars = ax6.bar(x_pos, times, width=0.5, color='#60BD68', edgecolor='black')
        add_value_labels(ax6)
        
        ax6.set_ylabel('Time (seconds)', fontsize=12)
        ax6.set_title(f'Execution Time\n({world_name})', fontsize=14)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(alg_names, rotation=30, ha='right')
        ax6.text(0.5, -0.25, "Lower is better - wall-clock efficiency", transform=ax6.transAxes, 
                ha='center', fontsize=8, style='italic')
        
        # Save the plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for main title
        plt.savefig(f'{world_name.replace(" ", "_").lower()}_comparison.png', dpi=300, bbox_inches='tight')

        # Use a more compatible layout approach that avoids the tight_layout warnings
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.3)
        plt.savefig(f'{world_name.replace(" ", "_").lower()}_comparison.png', dpi=300, bbox_inches='tight')
    
    # Create a final summary figure
    plt.figure(figsize=(18, 6), clear=True)
    fig = plt.gcf()
    fig.clf()  # Clear any previous content
    fig.suptitle('Algorithm Performance Comparison Across All Environments', fontsize=18)
    # fig.text(0.5, 0.94, "Comparing Policy Iteration, Synchronous Value Iteration, and Asynchronous Value Iteration", 
    #          ha='center', fontsize=12, style='italic')
    
    axes = [plt.subplot(1, 3, i+1) for i in range(3)]
    
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
    axes[0].set_title('Iterations to Convergence\nAcross Environments', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(world_names, rotation=0)
    axes[0].legend(loc='upper left')
    axes[0].text(0.5, -0.15, "Lower is better", transform=axes[0].transAxes, 
                ha='center', fontsize=9, style='italic')
    
    # Plot Bellman updates comparison
    for i, alg in enumerate(alg_names):
        updates = [all_results[world][alg]['operations']['bellman_updates'] for world in world_names]
        bars = axes[1].bar(x + (i-1)*width, updates, width, label=alg)
    
    axes[1].set_ylabel('Bellman Updates', fontsize=12)
    axes[1].set_title('Total Bellman Updates\nComputational Workload', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(world_names, rotation=0)
    axes[1].legend(loc='upper left')
    axes[1].text(0.5, -0.15, "Lower is better", transform=axes[1].transAxes, 
                ha='center', fontsize=9, style='italic')
    
    # Plot execution time comparison
    for i, alg in enumerate(alg_names):
        times = [all_results[world][alg]['time'] for world in world_names]
        bars = axes[2].bar(x + (i-1)*width, times, width, label=alg)
    
    axes[2].set_ylabel('Time (seconds)', fontsize=12)
    axes[2].set_title('Execution Time\nWall-Clock Performance', fontsize=14)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(world_names, rotation=0)
    axes[2].legend(loc='upper left')
    axes[2].text(0.5, -0.15, "Lower is better", transform=axes[2].transAxes, 
                ha='center', fontsize=9, style='italic')

    # Create a final summary figure
    plt.figure(figsize=(18, 6), clear=True)
    fig = plt.gcf()
    fig.clf()  # Clear any previous content
    fig.suptitle('Algorithm Performance Comparison Across All Environments', fontsize=18)
    # fig.text(0.5, 0.94, "Comparing Policy Iteration, Synchronous Value Iteration, and Asynchronous Value Iteration", 
    #          ha='center', fontsize=12, style='italic')
    
    # Create distinct colors for each algorithm for better visibility
    colors = ['#3366CC', '#DC3912', '#FF9900']
    
    # Create three subplots in a row
    axes = [plt.subplot(1, 3, i+1) for i in range(3)]
    
    # Set up data for comparison across worlds
    world_names = list(worlds.keys())
    alg_names = list(all_results[world_names[0]].keys())
    x = np.arange(len(world_names))
    width = 0.25
    
    # Plot iterations comparison with value labels
    for i, alg in enumerate(alg_names):
        iterations = [all_results[world][alg]['operations']['iterations'] for world in world_names]
        bars = axes[0].bar(x + (i-1)*width, iterations, width, label=alg, color=colors[i])
        
        # Add value labels on top of each bar
        for j, val in enumerate(iterations):
            axes[0].text(x[j] + (i-1)*width, val + 0.5, f"{val}", 
                        ha='center', va='bottom', fontsize=8, rotation=0)
    
    axes[0].set_ylabel('Number of Iterations', fontsize=12)
    axes[0].set_title('Iterations to Convergence\nAcross Environments', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(world_names, rotation=0)
    axes[0].legend(loc='upper left', title="Algorithm")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].text(0.5, -0.15, "Lower is better - fewer iterations needed", transform=axes[0].transAxes, 
                ha='center', fontsize=9, style='italic')
    
    # Plot Bellman updates comparison with value labels
    for i, alg in enumerate(alg_names):
        updates = [all_results[world][alg]['operations']['bellman_updates'] for world in world_names]
        bars = axes[1].bar(x + (i-1)*width, updates, width, label=alg, color=colors[i])
        
        # Add value labels on top of each bar
        for j, val in enumerate(updates):
            axes[1].text(x[j] + (i-1)*width, val + 5, f"{val}", 
                        ha='center', va='bottom', fontsize=8, rotation=0)
    
    axes[1].set_ylabel('Number of Updates', fontsize=12)
    axes[1].set_title('Total Bellman Updates\nComputational Workload', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(world_names, rotation=0)
    axes[1].legend(loc='upper left', title="Algorithm")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].text(0.5, -0.15, "Lower is better - fewer computations", transform=axes[1].transAxes, 
                ha='center', fontsize=9, style='italic')
    
    # Plot execution time comparison with value labels
    for i, alg in enumerate(alg_names):
        times = [all_results[world][alg]['time'] for world in world_names]
        bars = axes[2].bar(x + (i-1)*width, times, width, label=alg, color=colors[i])
        
        # Add value labels on top of each bar
        for j, val in enumerate(times):
            axes[2].text(x[j] + (i-1)*width, val + 0.01, f"{val:.3f}s", 
                        ha='center', va='bottom', fontsize=8, rotation=0)
    
    axes[2].set_ylabel('Time (seconds)', fontsize=12)
    axes[2].set_title('Execution Time\nWall-Clock Performance', fontsize=14)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(world_names, rotation=0)
    axes[2].legend(loc='upper left', title="Algorithm")
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    axes[2].text(0.5, -0.15, "Lower is better - faster runtime", transform=axes[2].transAxes, 
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust for main title
    plt.savefig('overall_comparison.png', dpi=300, bbox_inches='tight')

    # Use a more compatible layout approach that avoids the tight_layout warnings
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.3)
    plt.savefig('overall_comparison.png', dpi=300, bbox_inches='tight')
    
    # Prevent showing extra plots
    plt.close('all') 