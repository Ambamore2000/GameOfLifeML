import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('gameoflifeml/data/GLIDER_optimal_model.h5')

# Assuming next_generation function from gol.py is imported
from gameoflifeml.generate.gol import next_generation

def visualize_sequence(start_grid, steps):
    """
    Visualizes the sequence of Conway's Game of Life states, model predictions, and actual next states.

    Parameters:
    - start_grid (np.array): Initial state of the game grid.
    - steps (int): Number of steps to simulate.
    """
    current_grid = start_grid.copy()
    fig, axes = plt.subplots(steps, 3, figsize=(10, steps * 3))

    for i in range(steps):
        # Current state
        axes[i, 0].imshow(current_grid, cmap='binary')
        axes[i, 0].set_title(f"Step {i}: Current State")
        axes[i, 0].axis('off')

        # Predicted next state
        predicted_next_state = model.predict(np.expand_dims(np.expand_dims(current_grid, axis=0), axis=-1))[0]
        predicted_next_state = (predicted_next_state > 0.5).astype(int)
        axes[i, 1].imshow(predicted_next_state.squeeze(), cmap='binary')
        axes[i, 1].set_title(f"Step {i}: Model's Prediction")
        axes[i, 1].axis('off')

        # Actual next state
        next_grid = next_generation(current_grid)
        axes[i, 2].imshow(next_grid, cmap='binary')
        axes[i, 2].set_title(f"Step {i}: Actual Next State")
        axes[i, 2].axis('off')

        # Update current grid for the next iteration
        current_grid = next_grid

    plt.tight_layout()
    plt.show()

def animate_sequence(start_grid, steps):
    """
    Creates an animation showing the game's progress, model's predictions, and actual next states,
    and loops the animation.

    Parameters:
    - start_grid (np.array): Initial state of the game grid.
    - steps (int): Number of steps to simulate.
    """
    current_grid = start_grid.copy()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    def update(frame):
        # Update plots for current, predicted, and actual next states
        for ax in axes:
            ax.clear()

        # Current state
        axes[0].imshow(current_grid, cmap='binary')
        axes[0].set_title("Current State")
        axes[0].axis('off')

        # Predicted next state
        predicted_next_state = model.predict(np.expand_dims(np.expand_dims(current_grid, axis=0), axis=-1))[0]
        predicted_next_state = (predicted_next_state > 0.5).astype(int)
        axes[1].imshow(predicted_next_state.squeeze(), cmap='binary')
        axes[1].set_title("Model's Prediction")
        axes[1].axis('off')

        # Actual next state
        next_grid = next_generation(current_grid)
        axes[2].imshow(next_grid, cmap='binary')
        axes[2].set_title("Actual Next State")
        axes[2].axis('off')

        # Update current grid for the next frame
        current_grid[:] = next_grid

    ani = animation.FuncAnimation(fig, update, frames=steps, repeat=True)
    plt.show()

    return ani


if __name__ == "__main__":
    start_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ])

    # visualize_sequence(start_grid, 10)  # Change the number of steps as needed
    # Or, to create an animation:
    animate_sequence(start_grid, 10)
