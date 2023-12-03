import numpy as np

def next_generation(grid):
    """
    Compute the next generation of cells for Conway's Game of Life.

    Parameters:
    grid (np.array): A 2D array of integers representing the current state of the game board.

    Returns:
    np.array: A 2D array representing the next state of the game board.
    """
    neighbors = sum(np.roll(np.roll(grid, i, 0), j, 1)
                    for i in (-1, 0, 1) for j in (-1, 0, 1)
                    if (i != 0 or j != 0))
    return (neighbors == 3) | ((grid == 1) & (neighbors == 2))

def generate_glider_data(size, steps):
    """
    Generate data representing different stages of a glider in Conway's Game of Life.

    Parameters:
    size (int): The size of the game board (size x size).
    steps (int): Number of evolution steps to simulate.

    Returns:
    np.array: A 3D array containing the evolution of the glider pattern over the specified steps.
    """
    data = []
    grid = np.zeros((size, size), dtype=int)
    grid[:3, :3] = [[0, 1, 0], [0, 0, 1], [1, 1, 1]]  # Initial glider pattern

    for _ in range(steps):
        data.append(grid.copy())
        grid = next_generation(grid)
    
    return np.array(data)

if __name__ == "__main__":
    size = 10
    steps = 20
    glider_data = generate_glider_data(size, steps)
