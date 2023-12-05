import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow import keras
import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    # Flatten the tensors
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
    # Compute Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# Load the trained model
model = keras.models.load_model('gameoflifeml/data/gameoflife_cnn_model.h5', 
                                custom_objects={'dice_coefficient': dice_coefficient})
# Assuming next_generation function from gol.py is imported
from gameoflifeml.generate.gol import next_generation
from gameoflifeml.train.trainer import create_random_glider

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

        # Set borders for each subplot
        for j in range(3):
            for spine in axes[i, j].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(10)

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

        # Set borders for each subplot
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(10)

        # Update current grid for the next frame
        current_grid[:] = next_grid

    ani = animation.FuncAnimation(fig, update, frames=steps, repeat=True)
    plt.show()

    return ani


if __name__ == "__main__":
    test_grid = create_random_glider(100)


    # visualize_sequence(start_grid, 10)  # Change the number of steps as needed
    # Or, to create an animation:
    animate_sequence(test_grid, 10)