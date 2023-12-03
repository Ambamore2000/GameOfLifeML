# config.py for the gameoflifeml project

import os

# MODEL_TYPE: A string variable to define the type of model configuration to be used.
# It can be set externally via an environment variable. If not set, it defaults to 'optimal'.
# Possible values: 'low_end', 'optimal', 'high_end'.
MODEL_TYPE = os.getenv('MODEL_TYPE', 'optimal')

# MODEL_CONFIGS: A dictionary that maps each MODEL_TYPE to its specific configuration.
# Each configuration is a dictionary with keys 'SIZE', 'STEPS', and 'NUM_SAMPLES'.
# - 'low_end': Configuration for low-end model, less resource-intensive
# - 'optimal': Balanced configuration, intended as a default setting
# - 'high_end': Configuration for high-end model, more resource-intensive
MODEL_CONFIGS = {
    'low_end': {'SIZE': 10, 'STEPS': 5, 'NUM_SAMPLES': 500},
    'optimal': {'SIZE': 20, 'STEPS': 10, 'NUM_SAMPLES': 2000},
    'high_end': {'SIZE': 30, 'STEPS': 15, 'NUM_SAMPLES': 5000}
}

# current_config: Retrieves the specific configuration for the current MODEL_TYPE.
# Uses the .get() method for safe access to the MODEL_CONFIGS dictionary.
# If MODEL_TYPE is not a key in MODEL_CONFIGS, defaults to the 'optimal' configuration.
current_config = MODEL_CONFIGS.get(MODEL_TYPE, MODEL_CONFIGS['optimal'])

# SIZE, STEPS, NUM_SAMPLES: Extracted specific settings from the current configuration.
# These settings can be imported and used across various modules in the project.
SIZE = current_config['SIZE']
STEPS = current_config['STEPS']
NUM_SAMPLES = current_config['NUM_SAMPLES']
