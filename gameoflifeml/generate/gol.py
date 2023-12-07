import numpy as np
from tqdm import tqdm
import random

def next_generation(grid):
    neighbors = sum(np.roll(np.roll(grid, i, 0), j, 1)
                    for i in (-1, 0, 1) for j in (-1, 0, 1)
                    if (i != 0 or j != 0))
    return ((neighbors == 3) | ((grid == 1) & (neighbors == 2))).astype(int)

def place_pattern(grid, pattern, position):
    x, y = position
    size_x, size_y = pattern.shape
    grid[x:x+size_x, y:y+size_y] = pattern
    return grid

def random_rotation(pattern):
    num_rotations = random.choice([1, 2, 3])
    return np.rot90(pattern, num_rotations)

def random_flip(pattern):
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

def generate_random_data(size, steps, num_samples):
    data = []

    for _ in tqdm(range(num_samples), desc="Generating Data"):
        grid = np.random.randint(2, size=(size, size), dtype=int)

        sequence = [grid.copy()]

        for _ in range(steps - 1):
            grid = next_generation(grid)
            sequence.append(grid.copy())

        data.append(sequence)

    return np.array(data).reshape(num_samples, steps, size, size)