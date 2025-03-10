import matplotlib.pyplot as plt

import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.world_config import small_world
from environment.model import Model
from algorithms.q_learning import run_q_learning_vs_sarsa_comparison, hyperparameter_tuning_q_learning

if __name__ == "__main__":
    # Create model using small_world configuration
    model = Model(small_world)
    
    print("===== Q-Learning vs SARSA Comparison =====")
    print(f"World: Small World ({small_world.num_rows}x{small_world.num_cols})")
    print(f"Start: {small_world.start_cell}, Goal: {small_world.goal_cell}")
    print(f"Obstacles: {small_world.obstacle_cells}")
    print(f"Discount factor (gamma): {model.gamma}")
    print(f"Transition probabilities: p_good={small_world.prob_good_trans}, bias={small_world.bias}")
    print(f"Rewards: step={small_world.reward_step}, goal={small_world.reward_goal}")
    print()
    
    # Option 1: Run hyperparameter tuning first to find best params
    print("Running hyperparameter tuning for Q-learning...")
    best_alpha, best_epsilon = hyperparameter_tuning_q_learning(model)
    
    # Option 2: Run comparison with selected parameters
    print("\nRunning Q-learning vs SARSA comparison...")
    results = run_q_learning_vs_sarsa_comparison(
        model=model,
        num_episodes=500,
        max_steps=1000,
        alpha=best_alpha,  # Using best parameters from tuning
        epsilon=best_epsilon,
        epsilon_decay=0.001,
        plot_save_name="small_world_comparison"
    )
    
    # Show the plots
    plt.show() 