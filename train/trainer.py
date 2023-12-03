import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_generation import generate_glider_data

def create_model(input_size):
    """
    Create a simple neural network model for Conway's Game of Life.

    Parameters:
    input_size (int): The size of the input layer.

    Returns:
    tf.keras.Model: The compiled neural network model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size, input_size)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(input_size * input_size, activation='sigmoid'),
        tf.keras.layers.Reshape((input_size, input_size))
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train the neural network model.

    Parameters:
    model (tf.keras.Model): The neural network model.
    X_train, y_train (np.array): Training data and labels.
    X_test, y_test (np.array): Testing data and labels.

    Returns:
    tf.keras.callbacks.History: The history object containing training information.
    """
    return model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

if __name__ == "__main__":
    size = 10
    steps = 20
    glider_data = generate_glider_data(size, steps)
    X_train, X_test, y_train, y_test = train_test_split(glider_data, glider_data, test_size=0.2)
    model = create_model(size)
    history = train_model(model, X_train, y_train, X_test, y_test)
