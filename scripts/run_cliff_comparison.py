"""
Run comparison of SARSA and Q-learning on the Cliff World environment.

This script produces:
1. Learning curves showing episode rewards for both algorithms
2. Policy visualizations showing the different paths
3. Analysis of performance differences
"""

import matplotlib.pyplot as plt
import numpy as np
from cliff_world_comparison import run_cliff_world_comparison, analyze_path_safety
from model import Model
from world_config import cliff_world

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("===== Comparing SARSA and Q-learning on Cliff World =====")
    print("This will reproduce analysis similar to Figure 6.13 in Sutton & Barto")
    
    # Run comparison with parameters that highlight the differences
    results = run_cliff_world_comparison(
        num_episodes=500,      # Train for 500 episodes
        max_steps=200,         # Cap episode length at 200 steps
        alpha=0.1,             # Learning rate
        epsilon=0.1,           # Exploration rate 
        epsilon_decay=0.001,   # Gradually reduce exploration
        smooth_window=10,      # Window size for smoothing reward curves
        verbose=True,          # Print progress and results
        plot_save_name="cliff_world_comparison"
    )
    
    # Analyze policy paths and safety
    model = Model(cliff_world)
    bad_states = [model.cell2state(cell) for cell in cliff_world.bad_cells]
    
    print("\n===== Path Safety Analysis =====")
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
    
    print("\n===== Conclusion =====")
    print("This experiment demonstrates a key difference between SARSA and Q-learning in cliff world:")
    print(" - SARSA (on-policy) learns a policy accounting for the exploration strategy")
    print("   it's using, leading to a safer path further from the cliff edge.")
    print(" - Q-learning (off-policy) learns the optimal deterministic policy,")
    print("   which often takes a shorter path closer to the cliff.")
    print("")
    print("The episode rewards reflect this difference:")
    print(" - SARSA typically has more stable returns once it learns a safe policy")
    print(" - Q-learning may have occasional large negative returns due to")
    print("   exploratory actions causing the agent to fall off the cliff")
    print("")
    print("This reproduces a classic result from Sutton & Barto's textbook,")
    print("showing how on-policy vs. off-policy learning affects risk-sensitive behaviors.")
    
    # Show plots
    plt.show() 