import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras import backend as K
from gameoflifeml.generate.gol import place_pattern   # Adjust the import path as needed
from .. import config

# Define the Dice coefficient for use as a metric
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Define the model architecture
def residual_block(x, filters, kernel_size=3):
    """A residual block for the CNN"""
    y = Conv2D(filters, kernel_size, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters, kernel_size, padding='same')(y)
    y = BatchNormalization()(y)
    out = Add()([x, y])
    out = Activation('relu')(out)
    return out

def game_of_life_model(input_size=(100, 100, 1)):
    inputs = Input(input_size)
    x = Conv2D(64, 3, activation='relu', padding='same')(inputs)

    # Add residual blocks
    for _ in range(4):
        x = residual_block(x, 64)

    output = Conv2D(1, 1, activation='sigmoid')(x)
    model = Model(inputs, output)
    # Compile the model with additional metrics
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', 
                  metrics=['accuracy', Precision(), Recall(), AUC(), dice_coefficient])
    return model

# Data loading function
def load_data(data_path):
    data = np.load(data_path)
    input_frames = data[:, 0, :, :].reshape(-1, 100, 100, 1)
    target_frames = data[:, 1, :, :].reshape(-1, 100, 100, 1)
    return input_frames, target_frames

# Create a random glider pattern for testing
def create_random_glider(size):
    glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])
    grid = np.zeros((size, size), dtype=int)
    x, y = np.random.randint(0, size - 3, size=2)
    place_pattern(grid, glider, (x, y))
    return grid

# Main execution
if __name__ == "__main__":
    # Path to your training data
    data_path = config.DATA_PATH + "RANDOM_optimal.npy" # Adjust this path

    # Load data
    input_frames, target_frames = load_data(data_path)
    print(f"Input Frames Shape: {input_frames.shape}")
    print(f"Target Frames Shape: {target_frames.shape}")

    # Initialize and train the model
    model = game_of_life_model()
    model.fit(input_frames, target_frames, batch_size=16, epochs=3, validation_split=0.2)

    # Evaluate the model
    evaluation = model.evaluate(input_frames, target_frames)
    print("Evaluation Metrics:")
    print("Loss:", evaluation[0])
    print("Accuracy:", evaluation[1])
    print("Precision:", evaluation[2])
    print("Recall:", evaluation[3])
    print("AUC:", evaluation[4])
    print("Dice Coefficient:", evaluation[5])

    # Calculate F1-Score manually
    precision = evaluation[2]
    recall = evaluation[3]
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    print("F1-Score:", f1_score)
    # Save the model
    model.save(config.DATA_PATH + 'gameoflife_cnn_model.h5')

    # Test the model with a random glider pattern
    test_grid = create_random_glider(100)
    predicted_frame = model.predict(np.expand_dims(test_grid, axis=0))
    predicted_frame_binary = (predicted_frame > 0.5).astype(int)
    print("Predicted frame:")
    for row in predicted_frame_binary[0, :, :, 0]:
        print(' '.join(map(str, row)))