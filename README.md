# Reinforcement Learning in Grid-World Environments

This project explores **Reinforcement Learning (RL)** algorithms applied to various **grid-world** scenarios. The codebase demonstrates how different methods—**Policy Iteration (PI)**, **Value Iteration (VI)**, **SARSA**, **Expected SARSA**, **Q-learning**, and **function approximation**—can be implemented, tuned, and compared.  

## Overview of the Project

1. **Grid-World Environments**  
   - **Small World**: A 4×4 grid with a few obstacles.  
   - **Grid World**: A larger grid with multiple obstacles.  
   - **Cliff World**: A 5×10 environment with a cliff penalty.  
   - **Dynamic Small World**: A variant where certain barrier cells are randomly activated at the start of each episode.

2. **Algorithms Implemented**  
   - **Policy Iteration** (Dynamic Programming)  
   - **Value Iteration** (Synchronous and Asynchronous)  
   - **SARSA** (On-policy TD control)  
   - **Expected SARSA** (On-policy TD control using expected returns)  
   - **Q-learning** (Off-policy TD control)  
   - **Function Approximation** (Linear and DQN) for handling larger or dynamic state spaces

3. **Key Features**  
   - **Stopping Criteria**: For Value Iteration, a threshold \(\epsilon\) for convergence.  
   - **Hyperparameter Tuning**: Automatic testing of different \(\alpha\) (learning rate) and \(\epsilon\) (exploration) values to find optimal settings.  
   - **Performance Metrics**:  
     - Number of iterations or episodes to converge  
     - Bellman updates  
     - Execution time  
     - Memory usage (for tabular vs. function approximation)  
     - Episode returns (e.g., total reward per episode)  

4. **Notable Results**  
   - **Value Iteration vs. Policy Iteration**: Both converge to an optimal policy, but VI may require more iterations; asynchronous updates often speed convergence.  
   - **SARSA vs. Q-learning**: Q-learning can yield higher returns off-policy but may be riskier, as shown in **Cliff World**.  
   - **Expected SARSA**: Sometimes more stable than SARSA.  
   - **Function Approximation**: Useful when the state space grows (e.g., random barriers). Linear approximation can be efficient and sufficient for moderate complexity, whereas DQNs require careful tuning.

5. **How to Run**  
   - **Dependencies**: Python 3.x, NumPy, Matplotlib, possibly PyTorch or TensorFlow if you use the DQN.  
   - **Scripts**:  
     - `policy_iteration.py`, `value_iteration.py`: Demonstrate DP methods.  
     - `sarsa.py`, `q_learning.py`: Implement SARSA and Q-learning.  
     - `compare_algorithms.py`: Compares PI and VI across different worlds.  
     - `run_cliff_comparison.py`: Plots SARSA vs. Q-learning returns in **Cliff World**.  
     - `function_approximation.py`: Implements tabular, linear, and DQN agents in the **Dynamic Small World**.  
   - **Usage**:  
     ```bash
     python compare_algorithms.py
     python run_cliff_comparison.py
     python function_approximation.py
     ```
     Each script logs results and generates plots in the `images/` directory.

6. **Project Structure**  
   - **`model.py`** & **`world_config.py`**: Define the MDP (states, actions, transitions, rewards).  
   - **`plot_vp.py`**: Visualization for value functions and policies.  
   - **`sarsa.py`, `q_learning.py`**: Implement on-policy and off-policy TD control.  
   - **`function_approximation.py`**: Showcases how linear or DQN methods can handle a dynamic environment.  
   - **`compare_algorithms.py`**: Runs Value Iteration vs. Policy Iteration.  
   - **`run_cliff_comparison.py`**: Compares SARSA and Q-learning in Cliff World.  

7. **Extensions**  
   - Add more advanced function approximators (e.g., deeper networks, experience replay).  
   - Explore different exploration strategies (e.g., Boltzmann exploration).  
   - Integrate policy gradient methods for continuous action spaces.