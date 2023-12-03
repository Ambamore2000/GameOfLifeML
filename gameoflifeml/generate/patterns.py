"""
patterns.py

This module defines a set of constant patterns used in Conway's Game of Life, 
along with a function to generate random patterns.

Each predefined pattern is represented as a NumPy array where 0 indicates a dead cell 
and 1 indicates a live cell. These patterns are commonly used as initial configurations 
in the Game of Life simulations. The module also includes a utility function to generate 
random patterns of a specified size.

Patterns:
- GLIDER: A small pattern that moves across the board.
- BLINKER: A simple oscillator with a period of 2.
- TOAD: Another small oscillator with a period of 2.
- BEACON: A pattern with a period of 2 that simulates two blocks interacting.

Functions:
- generate_random_pattern(size): Generates a random pattern of a given size.

Note: These patterns are defined as constants and should not be modified.

Author: Andrew Kim
Created: 12/3/23
"""

import numpy as np

# Glider pattern
GLIDER = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
])

# Blinker pattern
BLINKER = np.array([
    [1, 1, 1]
])

# Toad pattern
TOAD = np.array([
    [0, 1, 1, 1],
    [1, 1, 1, 0]
])

# Beacon pattern
BEACON = np.array([
    [1, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1]
])

def generate_random_pattern(size):
    """
    Generate a random pattern for Conway's Game of Life.

    Parameters:
    - size (tuple): A tuple (width, height) representing the size of the grid.

    Returns:
    - np.array: A 2D NumPy array representing the random pattern.
    """
    width, height = size
    pattern = np.random.choice([0, 1], size=(width, height))
    return pattern