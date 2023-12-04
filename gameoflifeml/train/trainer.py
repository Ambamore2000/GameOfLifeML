import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Define the grid size
size = 5

# Load the data from the .npy file generated by generator.py
data_path = 'gameoflifeml/data/GLIDER_optimal.npy'  # Replace with the correct path to your .npy file
data = np.load(data_path)

# Print the frame sizes
print("Loaded data shape:", data.shape)

# Split the data into input and target frames with alternating frames
input_frames = data[0:len(data) - 1]  # Select every other frame, starting from the first
target_frames = data[1:len(data)]  # Select every other frame, starting from the second

# Print the frame sizes after padding
print("Input frames shape:", input_frames.shape)
print("Target frames shape:", target_frames.shape)

print(input_frames)
#exit()

# Define a simple convolutional neural network (CNN) model
model = keras.Sequential([
    layers.Input(shape=(size, size, 1)),  # Adjust 'size' based on your grid size
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
])

# Compile the model with additional metrics (accuracy, precision, recall, AUC)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()])

# Train the model
history = model.fit(input_frames, target_frames, epochs=20, batch_size=32)

# Evaluate the model on the entire dataset
evaluation = model.evaluate(input_frames, target_frames)
print("Evaluation Metrics:")
print("Loss:", evaluation[0])
print("Accuracy:", evaluation[1])
print("Precision:", evaluation[2])
print("Recall:", evaluation[3])
print("AUC:", evaluation[4])

# Calculate F1-Score manually
precision = evaluation[2]
recall = evaluation[3]
f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
print("F1-Score:", f1_score)

# Test the model on a new input grid to predict the next step
test_input_grid = np.array([[[0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]]], dtype=np.float32)

# Predict the next step
predicted_frame = model.predict(test_input_grid)
predicted_frame_binary = (predicted_frame > 0.5).astype(int)

print("Predicted frame:")
for row in predicted_frame_binary[0, :, :, 0]:
    print(' '.join(map(str, row)))