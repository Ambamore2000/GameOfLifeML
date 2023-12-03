import matplotlib.pyplot as plt
from data_generation import generate_glider_data
from train_model import create_model
import numpy as np

def plot_grid(grid, title="Game of Life"):
    """
    Plot a single grid of the Game of Life.

    Parameters:
    grid (np.array): A 2D array representing the game board.
    title (str): The title for the plot.
    """
    plt.imshow(grid, cmap='binary')
    plt.title(title)

def visualize_predictions(model, data, num_samples=5):
    """
    Visualize the actual and predicted states of the Game of Life grid.

    Parameters:
    model (tf.keras.Model): The trained neural network model.
    data (np.array): The dataset containing the actual states.
    num_samples (int): The number of samples to visualize.
    """
    for i in range(num_samples):
        actual = data[i]
        predicted = model.predict(actual.reshape(1, -1)).reshape(actual.shape)
        predicted = (predicted > 0.5).astype(int)  # Binarize the output

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plot_grid(actual, "Actual State")

        plt.subplot(1, 2, 2)
        plot_grid(predicted, "Predicted State")
        plt.show()

if __name__ == "__main__":
    size = 10
    steps = 20
    glider_data = generate_glider_data(size, steps)
    model = create_model(size)
    visualize_predictions(model, glider_data)
