import numpy as np
import random
from typing import List, Tuple, Dict, Set, NamedTuple, Optional
from collections import namedtuple

from world_config import WorldConfig, Cell, small_world
from model import Model, Actions, State

# Define a state type that includes barrier configuration
EnhancedState = namedtuple('EnhancedState', ['row', 'col', 'barriers'])

class RandomBarrierWorld:
    """
    An extension of the small_world environment where barriers can be randomly
    active or inactive at the start of each episode.
    
    This creates a much larger effective state space as the agent needs to track
    which cells are blocked in the current episode.
    """
    
    def __init__(self, base_config: WorldConfig = small_world, barrier_prob: float = 0.5):
        """
        Initialize a RandomBarrierWorld environment.
        
        Args:
            base_config: Base world configuration to extend
            barrier_prob: Probability that each barrier is active (default: 0.5)
        """
        self.base_config = base_config
        self.barrier_prob = barrier_prob
        
        # Store original barrier positions for reference
        self.potential_barriers = list(base_config.obstacle_cells)
        
        # Count possible barrier configurations (2^n where n is number of potential barriers)
        self.num_barrier_configs = 2 ** len(self.potential_barriers)
        
        # Calculate total state space size
        self.num_rows = base_config.num_rows
        self.num_cols = base_config.num_cols
        self.base_state_size = self.num_rows * self.num_cols
        self.total_state_size = self.base_state_size * self.num_barrier_configs
        
        # Other configuration parameters
        self.start_cell = base_config.start_cell
        self.goal_cell = base_config.goal_cell
        self.reward_step = base_config.reward_step
        self.reward_goal = base_config.reward_goal
        self.reward_bad = base_config.reward_bad
        self.gamma = base_config.gamma
        self.prob_good_trans = base_config.prob_good_trans
        self.bias = base_config.bias
        
        # Current barrier configuration (will be reset at episode start)
        self.active_barriers: Set[Cell] = set()
        self.barrier_config: Tuple[bool, ...] = tuple()
        
        # Initialize barriers
        self.reset_barriers()
        
        print(f"RandomBarrierWorld initialized with:")
        print(f"  Base grid size: {self.num_rows}x{self.num_cols}")
        print(f"  Potential barriers: {len(self.potential_barriers)}")
        print(f"  Possible barrier configurations: {self.num_barrier_configs}")
        print(f"  Total state space size: {self.total_state_size}")
    
    def reset_barriers(self) -> Tuple[bool, ...]:
        """
        Randomly reset which barriers are active.
        
        Returns:
            Tuple of booleans representing which barriers are active
        """
        # Determine which barriers are active
        self.barrier_config = tuple(
            random.random() < self.barrier_prob 
            for _ in range(len(self.potential_barriers))
        )
        
        # Update active barriers set
        self.active_barriers = {
            barrier for i, barrier in enumerate(self.potential_barriers)
            if self.barrier_config[i]
        }
        
        return self.barrier_config
    
    def to_enhanced_state(self, cell: Cell) -> EnhancedState:
        """
        Convert a cell to an enhanced state that includes barrier configuration.
        
        Args:
            cell: The cell position (row, col)
            
        Returns:
            Enhanced state including barrier configuration
        """
        return EnhancedState(cell.row, cell.col, self.barrier_config)
    
    def is_valid_move(self, cell: Cell) -> bool:
        """
        Check if a cell is valid to move into.
        
        Args:
            cell: The cell to check
            
        Returns:
            True if the cell is valid, False otherwise
        """
        # Check if cell is within grid boundaries
        if not (0 <= cell.row < self.num_rows and 0 <= cell.col < self.num_cols):
            return False
        
        # Check if cell is an active barrier
        if cell in self.active_barriers:
            return False
        
        return True
    
    def get_next_cell(self, current_cell: Cell, action: Actions) -> Cell:
        """
        Get the next cell after applying an action, respecting barriers.
        
        Args:
            current_cell: Current cell position
            action: Action to take
            
        Returns:
            Next cell position
        """
        if action == Actions.UP:
            next_cell = Cell(current_cell.row - 1, current_cell.col)
        elif action == Actions.DOWN:
            next_cell = Cell(current_cell.row + 1, current_cell.col)
        elif action == Actions.LEFT:
            next_cell = Cell(current_cell.row, current_cell.col - 1)
        elif action == Actions.RIGHT:
            next_cell = Cell(current_cell.row, current_cell.col + 1)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Check if move is valid, otherwise stay in place
        if self.is_valid_move(next_cell):
            return next_cell
        else:
            return current_cell
    
    def get_reward(self, cell: Cell) -> float:
        """
        Get the reward for being in a cell.
        
        Args:
            cell: The cell position
            
        Returns:
            Reward value
        """
        if cell == self.goal_cell:
            return self.reward_goal
        else:
            return self.reward_step
    
    def is_terminal(self, cell: Cell) -> bool:
        """
        Check if a cell is terminal.
        
        Args:
            cell: The cell position
            
        Returns:
            True if the cell is terminal, False otherwise
        """
        return cell == self.goal_cell
    
    def get_transition_probabilities(self, current_cell: Cell, action: Actions) -> Dict[Cell, float]:
        """
        Get transition probabilities for an action, including stochasticity.
        
        Args:
            current_cell: Current cell position
            action: Action to take
            
        Returns:
            Dictionary mapping next cells to probabilities
        """
        # Define action rotation mapping
        left_action = {
            Actions.UP: Actions.LEFT,
            Actions.DOWN: Actions.RIGHT,
            Actions.LEFT: Actions.DOWN,
            Actions.RIGHT: Actions.UP,
        }
        
        right_action = {
            Actions.UP: Actions.RIGHT,
            Actions.DOWN: Actions.LEFT,
            Actions.LEFT: Actions.UP,
            Actions.RIGHT: Actions.DOWN,
        }
        
        # Calculate intended and actual next cells
        intended_cell = self.get_next_cell(current_cell, action)
        left_cell = self.get_next_cell(current_cell, left_action[action])
        right_cell = self.get_next_cell(current_cell, right_action[action])
        
        # Calculate transition probabilities
        transitions = {}
        transitions[intended_cell] = self.prob_good_trans
        transitions[left_cell] = (1 - self.prob_good_trans) * self.bias
        transitions[right_cell] = (1 - self.prob_good_trans) * (1 - self.bias)
        
        return transitions


class RandomBarrierModel(Model):
    """
    Extension of the Model class to handle RandomBarrierWorld environments.
    
    This model maintains a mapping between enhanced states (cell+barriers) and
    state indices to enable tabular methods to work.
    """
    
    def __init__(self, random_barrier_world: RandomBarrierWorld):
        """
        Initialize a model for the RandomBarrierWorld.
        
        Args:
            random_barrier_world: The RandomBarrierWorld environment
        """
        self.rbw = random_barrier_world
        self.gamma = random_barrier_world.gamma
        
        # Create a mapping from enhanced states to state indices
        self.state_mapping = {}
        self.reverse_mapping = {}
        
        state_idx = 0
        
        # For all possible cell positions
        for row in range(random_barrier_world.num_rows):
            for col in range(random_barrier_world.num_cols):
                # For all possible barrier configurations
                for config_idx in range(random_barrier_world.num_barrier_configs):
                    # Convert config index to binary configuration
                    barrier_config = self._index_to_barrier_config(
                        config_idx, len(random_barrier_world.potential_barriers)
                    )
                    
                    # Create enhanced state
                    enhanced_state = EnhancedState(row, col, barrier_config)
                    
                    # Add to mappings
                    self.state_mapping[enhanced_state] = state_idx
                    self.reverse_mapping[state_idx] = enhanced_state
                    
                    state_idx += 1
        
        # Create fictional end state
        self.num_states = state_idx + 1
        self.fictional_end_state = state_idx
        
        # Set start and goal states (depend on barrier configuration)
        self.start_cell = random_barrier_world.start_cell
        self.goal_cell = random_barrier_world.goal_cell
        
        # Track current barrier configuration
        self.current_barrier_config = random_barrier_world.barrier_config
        
        # Calculate start and goal states with current barriers
        self.start_state = self.enhanced_state_to_index(
            EnhancedState(
                self.start_cell.row, 
                self.start_cell.col, 
                self.current_barrier_config
            )
        )
        
        self.goal_state = self.enhanced_state_to_index(
            EnhancedState(
                self.goal_cell.row, 
                self.goal_cell.col, 
                self.current_barrier_config
            )
        )
        
        # Empty set for bad states (not used in small_world)
        self.bad_states = []
        self.obstacle_states = []  # This will be dynamically updated
        
        print(f"RandomBarrierModel initialized with {self.num_states} states")
    
    def _index_to_barrier_config(self, index: int, num_barriers: int) -> Tuple[bool, ...]:
        """
        Convert a configuration index to a barrier configuration tuple.
        
        Args:
            index: Index of the configuration
            num_barriers: Number of potential barriers
            
        Returns:
            Tuple of booleans representing which barriers are active
        """
        # Convert index to binary and pad with zeros
        binary = format(index, f"0{num_barriers}b")
        
        # Convert binary string to tuple of booleans
        return tuple(c == '1' for c in binary)
    
    def reset_environment(self) -> int:
        """
        Reset the environment by randomizing barriers and returning the new start state.
        
        Returns:
            The index of the new start state
        """
        # Reset barriers in the environment
        self.current_barrier_config = self.rbw.reset_barriers()
        
        # Update start state
        self.start_state = self.enhanced_state_to_index(
            EnhancedState(
                self.start_cell.row, 
                self.start_cell.col, 
                self.current_barrier_config
            )
        )
        
        # Update goal state
        self.goal_state = self.enhanced_state_to_index(
            EnhancedState(
                self.goal_cell.row, 
                self.goal_cell.col, 
                self.current_barrier_config
            )
        )
        
        # Update obstacle states based on active barriers
        self.obstacle_states = []
        for row in range(self.rbw.num_rows):
            for col in range(self.rbw.num_cols):
                cell = Cell(row, col)
                if cell in self.rbw.active_barriers:
                    self.obstacle_states.append(
                        self.enhanced_state_to_index(
                            EnhancedState(row, col, self.current_barrier_config)
                        )
                    )
        
        return self.start_state
    
    def enhanced_state_to_index(self, enhanced_state: EnhancedState) -> int:
        """
        Convert an enhanced state to its index.
        
        Args:
            enhanced_state: The enhanced state (row, col, barriers)
            
        Returns:
            Index of the state
        """
        return self.state_mapping.get(enhanced_state, self.fictional_end_state)
    
    def index_to_enhanced_state(self, index: int) -> EnhancedState:
        """
        Convert a state index to its enhanced state.
        
        Args:
            index: The state index
            
        Returns:
            Enhanced state (row, col, barriers)
        """
        if index == self.fictional_end_state:
            return None
        return self.reverse_mapping[index]
    
    def cell2state(self, cell: Cell) -> int:
        """
        Convert a cell to a state index using the current barrier configuration.
        
        Args:
            cell: The cell position
            
        Returns:
            Index of the state
        """
        enhanced_state = EnhancedState(
            cell.row, cell.col, self.current_barrier_config
        )
        return self.enhanced_state_to_index(enhanced_state)
    
    def state2cell(self, state: int) -> Cell:
        """
        Convert a state index to a cell.
        
        Args:
            state: The state index
            
        Returns:
            Cell position
        """
        if state == self.fictional_end_state:
            return None
        
        enhanced_state = self.index_to_enhanced_state(state)
        return Cell(enhanced_state.row, enhanced_state.col)
    
    def barrier_config_from_state(self, state: int) -> Tuple[bool, ...]:
        """
        Get the barrier configuration for a state.
        
        Args:
            state: The state index
            
        Returns:
            Tuple of booleans representing the barrier configuration
        """
        if state == self.fictional_end_state:
            return None
        
        enhanced_state = self.index_to_enhanced_state(state)
        return enhanced_state.barriers
    
    def reward(self, s: int, a: Actions) -> float:
        """
        Get the reward for taking an action in a state.
        
        Args:
            s: State index
            a: Action
            
        Returns:
            Reward value
        """
        if s == self.goal_state:
            return self.rbw.reward_goal
        elif s == self.fictional_end_state:
            return 0
        else:
            return self.rbw.reward_step
    
    def transition_probability(self, s1: int, s2: int, a: Actions) -> float:
        """
        Calculate transition probability from s1 to s2 given action a.
        
        Args:
            s1: Current state index
            s2: Next state index
            a: Action
            
        Returns:
            Transition probability
        """
        # Handle fictional end state
        if s1 == self.fictional_end_state:
            return 1.0 if s2 == self.fictional_end_state else 0.0
        
        # Handle obstacles and goal
        if s1 in self.obstacle_states or s1 == self.goal_state:
            return 1.0 if s2 == self.fictional_end_state else 0.0
        
        # Extract enhanced states
        es1 = self.index_to_enhanced_state(s1)
        es2 = self.index_to_enhanced_state(s2)
        
        # If the barrier configuration differs, transition is impossible
        if es1.barriers != es2.barriers:
            return 0.0
        
        # Get transition probabilities in cell space
        current_cell = Cell(es1.row, es1.col)
        next_cell = Cell(es2.row, es2.col)
        
        # Set active barriers in the environment to match the state
        original_barriers = self.rbw.active_barriers.copy()
        self.rbw.active_barriers = {
            barrier for i, barrier in enumerate(self.rbw.potential_barriers)
            if es1.barriers[i]
        }
        
        # Get transition probabilities
        cell_transitions = self.rbw.get_transition_probabilities(current_cell, a)
        
        # Restore original barriers
        self.rbw.active_barriers = original_barriers
        
        # Return probability of transitioning to the next cell
        return cell_transitions.get(next_cell, 0.0)


def create_random_barrier_world():
    """
    Create a random barrier world based on the small_world configuration.
    
    Returns:
        RandomBarrierWorld instance and RandomBarrierModel instance
    """
    # Create the random barrier world
    rbw = RandomBarrierWorld(small_world, barrier_prob=0.5)
    
    # Create the model
    model = RandomBarrierModel(rbw)
    
    return rbw, model


if __name__ == "__main__":
    # Test the random barrier world
    rbw, model = create_random_barrier_world()
    
    print(f"Random barrier world has {rbw.num_barrier_configs} possible barrier configurations")
    print(f"Total state space size: {rbw.total_state_size}")
    
    # Test resetting the environment
    start_state = model.reset_environment()
    print(f"Start state: {start_state}")
    print(f"Active barriers: {rbw.active_barriers}") 