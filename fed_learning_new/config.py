# -*- coding: utf-8 -*-
# Configuration constants for the Python 3.7 compatible FL setup.

import os

# --- Project Root Calculation ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Path Definitions ---
CONTRACT_DIR = os.path.join(PROJECT_ROOT, "contracts")
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "fl_py37_logs") # Log dir for this version
DEFAULT_CLIENT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "client_config.py")
DEFAULT_KEYSTORE_DIR = PROJECT_ROOT

# --- Contract Details ---
CONTRACT_NAME = "EnhancedFederatedLearning"
CONTRACT_FILENAME = f"{CONTRACT_NAME}.sol"
CONTRACT_PATH = os.path.join(CONTRACT_DIR, CONTRACT_FILENAME)
ABI_FILENAME = f"{CONTRACT_NAME}.abi"
BIN_FILENAME = f"{CONTRACT_NAME}.bin"
ABI_PATH = os.path.join(CONTRACT_DIR, ABI_FILENAME)
BIN_PATH = os.path.join(CONTRACT_DIR, BIN_FILENAME)
CONTRACT_NOTE_NAME = "federatedlearning_py37_demo" # Unique note name

# --- Default Federated Learning Parameters ---
DEFAULT_EPOCHS = 1
DEFAULT_ROUNDS = 2
DEFAULT_PARTICIPANTS = 2
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001

# --- Model Input/Output Dimensions (Defaults) ---
MNIST_INPUT_DIM_MLP = 28 * 28
MNIST_INPUT_CHANNELS_CNN = 1
MNIST_OUTPUT_DIM = 10

CIFAR_INPUT_DIM_MLP = 3 * 32 * 32
CIFAR_INPUT_CHANNELS_CNN = 3
CIFAR_OUTPUT_DIM = 10

# --- Other Constants ---
INITIAL_MODEL_STR = "Initial Model - Enhanced"
SERVER_ROLE_NAME = "server"