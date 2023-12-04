import numpy as np
import os
from .. import config
import matplotlib.pyplot as plt
from gameoflifeml.generate.patterns import GLIDER
from gameoflifeml.generate.gol import generate_data

def generate_training_data(name, input_grid, is_centered = False, is_transform = True):
    #generated_data = generate_data(config.SIZE, config.STEPS, config.NUM_SAMPLES, input_grid, is_centered, is_transform)
    generated_data = generate_data(5, 30, 100, input_grid, is_centered, is_transform)
    file_name = f"{name}_{config.MODEL_TYPE}"
#15, 'STEPS': 10, 'NUM_SAMPLES': 10
    #save_plots(name, generated_data)
    np.save(f"{config.DATA_PATH}{file_name}.npy", generated_data)

def save_plots(name, data):
    output_dir = os.path.join(config.DATA_PATH, config.MODEL_TYPE, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for step, grid in enumerate(data):
        group = step // config.STEPS
        plt.imshow(grid, cmap='binary')
        plt.title(f"Step {step}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"group_{group}_step_{step}.png"))
        plt.close()

if __name__ == "__main__":
    model_type = config.MODEL_TYPE

    generate_training_data("GLIDER", GLIDER, is_centered = True)