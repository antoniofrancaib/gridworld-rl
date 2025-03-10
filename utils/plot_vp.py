import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.model import Actions, Model
from environment.world_config import Cell


def plot_vp(model: Model, value_function: np.array, policy: np.array, ax=None, title=None):
    """
    Plot value function and policy on grid.

    :param model: The environment model
    :param value_function: 1D array of size `model.num_states` containing
        real values representing the value function at a given state.
    :param policy: 1D array of size `model.num_states` containing
        an action for each state.
    :param ax: Optional matplotlib axes to plot on. If None, a new figure is created.
    :param title: Optional title for the plot.
    
    :return: (fig, ax) tuple with figure and axes objects
    """
    # Check if we need to skip the fictional end state
    if len(value_function) == len(model.states):
        # Make a copy to avoid modifying the original
        v = value_function.copy()
        
        # Skip the fictional end state if needed
        if model.fictional_end_state in model.states:
            # If value_function is a dictionary
            if isinstance(v, dict):
                if model.fictional_end_state in v:
                    # Skip it in our plotting data
                    pass
            # If value_function is a numpy array
            else:
                # Ensure it's the right length (skip last element if needed)
                if len(v) > model.world.num_rows * model.world.num_cols:
                    v = v[:-1]  # get rid of final absorbing state
    else:
        # Already the right size
        v = value_function

    # Handle obstacles by setting them to -inf
    if isinstance(v, dict):
        # For dictionary value functions
        for cell in model.world.obstacle_cells:
            # cell is already a Cell object from WorldConfig
            cell_state = model.cell2state(cell)
            if cell_state in v:
                v[cell_state] = -np.inf
    else:
        # For array value functions
        for cell in model.world.obstacle_cells:
            # cell is already a Cell object from WorldConfig
            cell_state = model.cell2state(cell)
            if cell_state < len(v):
                v[cell_state] = -np.inf

    # Set up figure and axes
    scale = 1.2
    figsize = (scale * model.world.num_cols, scale * model.world.num_rows)
    
    if ax is None:
        # Create a new figure with a proper GridSpec that includes space for the colorbar
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(2, 1, height_ratios=[20, 1], figure=fig)
        
        # Create the main plot and colorbar axes using the GridSpec
        ax = fig.add_subplot(gs[0])
        cax = fig.add_subplot(gs[1])
    else:
        fig = ax.figure
        
        # If we're plotting on an existing axes in a GridSpec, 
        # we'll use a divider to create space for the colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.2)
    
    # Set up colormap
    cmap = mpl.cm.viridis
    cmap.set_bad("black", 1.0)
    
    # Reshape value function for plotting if it's an array
    if not isinstance(v, dict):
        plot_data = v.reshape(model.world.num_rows, model.world.num_cols)
    else:
        # Convert dictionary to 2D array
        plot_data = np.zeros((model.world.num_rows, model.world.num_cols))
        plot_data.fill(np.nan)
        for state, val in v.items():
            if state != model.fictional_end_state:
                cell = model.state2cell(state)
                # cell could be a tuple or Cell object depending on implementation
                if isinstance(cell, Cell):
                    plot_data[cell.row, cell.col] = val
                else:
                    plot_data[cell[0], cell[1]] = val
    
    # Plot the value function
    im = ax.matshow(plot_data, cmap=cmap)
    colorbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    colorbar.set_label('State Value')

    # Mark the start and goal states
    start_cell = model.world.start_cell
    goal_cell = model.world.goal_cell
    
    # Mark start state with a square marker
    ax.text(start_cell.col, start_cell.row, 'S', 
            ha='center', va='center', fontsize=14, 
            color='white', fontweight='bold')
    
    # Mark goal state with a star marker
    ax.text(goal_cell.col, goal_cell.row, 'G', 
            ha='center', va='center', fontsize=14,
            color='white', fontweight='bold')
    
    # Mark bad cells with 'B'
    for cell in model.world.bad_cells:
        ax.text(cell.col, cell.row, 'B',
                ha='center', va='center', fontsize=14,
                color='white', fontweight='bold')
    
    # Set up direction mappings for policy visualization
    U_LUT = {
        Actions.UP: 0,
        Actions.DOWN: 0,
        Actions.LEFT: -1,
        Actions.RIGHT: 1,
    }

    V_LUT = {
        Actions.UP: -1,  # Negative because row 0 is at the top
        Actions.DOWN: 1,
        Actions.LEFT: 0,
        Actions.RIGHT: 0,
    }

    # Create meshgrid for quiver plot
    cols, rows = np.meshgrid(
        range(model.world.num_cols), range(model.world.num_rows)
    )
    
    # Initialize arrow direction arrays
    U, V = np.zeros_like(cols), np.zeros_like(rows)
    
    # Populate arrow directions based on policy
    for r in range(model.world.num_rows):
        for c in range(model.world.num_cols):
            current_cell = Cell(r, c)
            
            # Skip obstacle cells
            if any(current_cell.row == cell.row and current_cell.col == cell.col 
                  for cell in model.world.obstacle_cells):
                continue
            
            # Skip start and goal cells
            if ((current_cell.row == model.world.start_cell.row and 
                 current_cell.col == model.world.start_cell.col) or
                (current_cell.row == model.world.goal_cell.row and 
                 current_cell.col == model.world.goal_cell.col)):
                continue
                
            # Skip bad cells
            if any(current_cell.row == cell.row and current_cell.col == cell.col 
                  for cell in model.world.bad_cells):
                continue
            
            # Get the state for this cell
            state = model.cell2state(current_cell)
                
            # Get action from policy
            if isinstance(policy, dict):
                if state in policy:
                    action = policy[state]
                else:
                    continue
            else:
                if state < len(policy):
                    action = policy[state]
                else:
                    continue
                
            # Set arrow directions
            U[r, c] = U_LUT[action]
            V[r, c] = V_LUT[action]

    # Plot policy arrows
    arrow_scale = 20
    ax.quiver(cols, rows, U, V, scale=arrow_scale, pivot='mid', color='white')

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, model.world.num_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, model.world.num_rows, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.2)
    
    # Adjust tick labels and grid appearance
    ax.set_xticks(np.arange(model.world.num_cols))
    ax.set_yticks(np.arange(model.world.num_rows))
    ax.set_xticklabels(np.arange(model.world.num_cols))
    ax.set_yticklabels(np.arange(model.world.num_rows))
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Value Function and Policy')

    return fig, ax
