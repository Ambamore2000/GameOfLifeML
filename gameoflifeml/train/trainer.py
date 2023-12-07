import os
from gameoflifeml.generate.gol import generate_random_data
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras import backend as K
from .. import config

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def residual_block(x, filters, kernel_size=3):
    y = Conv2D(filters, kernel_size, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters, kernel_size, padding='same')(y)
    y = BatchNormalization()(y)
    out = Add()([x, y])
    out = Activation('relu')(out)
    return out

def game_of_life_model(size):
    inputs = Input((size, size, 1))
    x = Conv2D(64, 3, activation='relu', padding='same')(inputs)

    for _ in range(4):
        x = residual_block(x, 64)

    output = Conv2D(1, 1, activation='sigmoid')(x)
    model = Model(inputs, output)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', 
                  metrics=['accuracy', Precision(), Recall(), AUC(), dice_coefficient])
    return model

def load_data(data):
    input_frames = data[:, 0, :, :].reshape(-1, 100, 100, 1)
    target_frames = data[:, 1, :, :].reshape(-1, 100, 100, 1)
    return input_frames, target_frames

def evaluate_gol_model(model_name, model, input_frames, target_frames):
    evaluation = model.evaluate(input_frames, target_frames)

    precision = evaluation[2]
    recall = evaluation[3]
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    evaluation_string = (
        "Evaluation Metrics:\n"
        f"Loss: {evaluation[0]}\n"
        f"Accuracy: {evaluation[1]}\n"
        f"Precision: {evaluation[2]}\n"
        f"Recall: {evaluation[3]}\n"
        f"AUC: {evaluation[4]}\n"
        f"Dice Coefficient: {evaluation[5]}\n"
        f"F1-Score: {f1_score}\n"
    )

    metrics_file_path = os.path.join(config.DATA_PATH, f'{model_name}_metrics.txt')

    with open(metrics_file_path, 'w') as file:
        file.write(evaluation_string)

def train_gol_model(size=100, steps=2, num_samples=10000, batch_size=16, epochs=3, validation_split=0.2):
    _model_name = f'gol_{num_samples}_{steps}_{size}_{size}_model'

    _generated_random_gol_data = generate_random_data(size, steps, num_samples)
    _input_frames, _target_frames = load_data(_generated_random_gol_data)

    _model = game_of_life_model(size)
    _model.fit(_input_frames, _target_frames, batch_size, epochs, validation_split)
    evaluate_gol_model(_model_name, _model, _input_frames, _target_frames)
    _model.save(config.DATA_PATH + f'{_model_name}.h5')    

def train_preset_gol_model(preset):
    train_gol_model(preset['SIZE'], 
                    preset['STEPS'], 
                    preset['NUM_SAMPLES'], 
                    preset['BATCH_SIZE'], 
                    preset['EPOCHS'], 
                    preset['VALIDATION_SPLIT'])

if __name__ == "__main__":
    MODEL_CONFIGS = {
        'optimal': {
            'SIZE': 100,
            'STEPS': 2,
            'NUM_SAMPLES': 10000,
            'BATCH_SIZE': 16,
            'EPOCHS': 2,
            'VALIDATION_SPLIT': 0.2
        },
        'low-end-test': {
            'SIZE': 100,
            'STEPS': 2,
            'NUM_SAMPLES': 1000,
            'BATCH_SIZE': 16,
            'EPOCHS': 5,
            'VALIDATION_SPLIT': 0.2
        }

    }

    train_preset_gol_model(MODEL_CONFIGS['optimal'])