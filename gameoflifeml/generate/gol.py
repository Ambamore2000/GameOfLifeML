"""
This module provides functionalities for generating and manipulating grids in 
Conway's Game of Life.

It includes functions for computing the next generation of a grid, placing a 
specific pattern on a grid, randomly rotating and flipping patterns, and 
generating a dataset of grid sequences with given patterns.

Functions:
- next_generation(grid): Computes the next generation of cells based on the 
  current state.
- place_pattern(grid, pattern, position): Places a specific pattern on the grid.
- random_rotation(pattern): Randomly rotates the pattern.
- random_flip(pattern): Randomly flips the pattern.
- generate_data(size, steps, num_samples, input_grid, is_centered): Generates a
  dataset of grid sequences with a given pattern.

Author: Andrew Kim
Created: 12/3/23
"""

import numpy as np
from tqdm import tqdm
import random

def next_generation(grid):
    """
    Compute the next generation of cells for Conway's Game of Life.

    Parameters:
    - grid (np.array): A 2D NumPy array representing the current state of the game board.

    Returns:
    - np.array: A 2D NumPy array representing the next state of the game board.
    """
    neighbors = sum(np.roll(np.roll(grid, i, 0), j, 1)
                    for i in (-1, 0, 1) for j in (-1, 0, 1)
                    if (i != 0 or j != 0))
    return ((neighbors == 3) | ((grid == 1) & (neighbors == 2))).astype(int)


def place_pattern(grid, pattern, position):
    """
    Place a specific pattern on the grid.

    Parameters:
    - grid (np.array): The grid on which to place the pattern.
    - pattern (np.array): The pattern to be placed.
    - position (tuple): The top-left position (row, column) where the pattern will be placed.

    Returns:
    - np.array: The updated grid with the pattern placed.
    """
    x, y = position
    size_x, size_y = pattern.shape
    grid[x:x+size_x, y:y+size_y] = pattern
    return grid

def random_rotation(pattern):
    """
    Randomly rotate the pattern by 90, 180, or 270 degrees.

    Parameters:
    - pattern (np.array): The pattern to be rotated.

    Returns:
    - np.array: The rotated pattern.
    """
    num_rotations = random.choice([1, 2, 3])
    return np.rot90(pattern, num_rotations)

def random_flip(pattern):
    """
    Randomly flip the pattern horizontally, vertically, or both.

    Parameters:
    - pattern (np.array): The pattern to be flipped.

    Returns:
    - np.array: The flipped pattern.
    """
    flip_horizontal = random.choice([True, False])
    flip_vertical = random.choice([True, False])
    if flip_horizontal and flip_vertical:
        return np.flipud(np.fliplr(pattern))
    elif flip_horizontal:
        return np.fliplr(pattern)
    elif flip_vertical:
        return np.flipud(pattern)
    else:
        return pattern

def generate_data(size, steps, num_samples, input_grid, is_centered=False, is_transform=False):
    """
    Generate a dataset of grid sequences for Conway's Game of Life.

    Parameters:
    - size (int): Size of the grid.
    - steps (int): Number of generations to simulate.
    - num_samples (int): Number of samples to generate.
    - input_grid (np.array): The initial pattern to be placed on the grid.
    - is_centered (bool): If True, the pattern will be centered in the grid.

    Returns:
    - np.array: A 3D NumPy array containing sequences of game states.
    """
    data = []

    for _ in tqdm(range(num_samples), desc="Generating Pattern Data"):
        
        if is_transform:
            input_grid = random_rotation(input_grid)
            input_grid = random_flip(input_grid)

        grid = np.zeros((size, size), dtype=int)

        if is_centered:
            x_offset = (size - input_grid.shape[0]) // 2
            y_offset = (size - input_grid.shape[1]) // 2
            position = (x_offset, y_offset)
        else:
            position = (np.random.randint(size - input_grid.shape[0] + 1), 
                        np.random.randint(size - input_grid.shape[1] + 1))

        grid[position[0]:position[0]+input_grid.shape[0], position[1]:position[1]+input_grid.shape[1]] = input_grid

        for _ in range(steps):
            data.append(grid.copy())
            grid = next_generation(grid)

    return np.array(data)