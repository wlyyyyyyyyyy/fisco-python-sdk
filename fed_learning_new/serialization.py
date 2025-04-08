# -*- coding: utf-8 -*-
# Model serialization/deserialization utilities. Python 3.7 compatible.

import io
import base64
import torch
import torch.nn as nn
from typing import Union # Use typing.Union

try:
    # Import models directly for creating instances
    from .models import MLP, CNN
    # Import config for constants and dimensions
    from .config import (
        INITIAL_MODEL_STR,
        MNIST_INPUT_DIM_MLP, MNIST_INPUT_CHANNELS_CNN, MNIST_OUTPUT_DIM,
        CIFAR_INPUT_DIM_MLP, CIFAR_INPUT_CHANNELS_CNN, CIFAR_OUTPUT_DIM
    )
except ImportError:
    # Fallback types for standalone testing
    class MLP(nn.Module): pass
    class CNN(nn.Module): pass
    INITIAL_MODEL_STR = "Initial Model - Enhanced"
    MNIST_INPUT_DIM_MLP, MNIST_INPUT_CHANNELS_CNN, MNIST_OUTPUT_DIM = 784, 1, 10
    CIFAR_INPUT_DIM_MLP, CIFAR_INPUT_CHANNELS_CNN, CIFAR_OUTPUT_DIM = 3072, 3, 10


def serialize_model(model: nn.Module) -> str:
    """Serializes a PyTorch model's state_dict to a Base64 string."""
    try:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        model_bytes = buffer.getvalue()
        return base64.b64encode(model_bytes).decode('utf-8')
    except Exception as e:
        print(f"[ERROR] Model serialization failed: {e}")
        return ""

def _create_new_model_instance(model_type: str, dataset_name: str) -> Union[nn.Module, None]: # Use Union
    """Helper to create a new model instance based on type and dataset."""
    try:
        if model_type == 'mlp':
            input_dim = MNIST_INPUT_DIM_MLP if dataset_name == 'mnist' else CIFAR_INPUT_DIM_MLP
            output_dim = MNIST_OUTPUT_DIM if dataset_name == 'mnist' else CIFAR_OUTPUT_DIM
            return MLP(input_dim=input_dim, output_dim=output_dim)
        elif model_type == 'cnn':
            input_channels = MNIST_INPUT_CHANNELS_CNN if dataset_name == 'mnist' else CIFAR_INPUT_CHANNELS_CNN
            output_dim = MNIST_OUTPUT_DIM if dataset_name == 'mnist' else CIFAR_OUTPUT_DIM
            return CNN(input_channels=input_channels, num_classes=output_dim)
        else:
            print(f"[ERROR] Unsupported model_type for instantiation: {model_type}")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to create model instance ({model_type}, {dataset_name}): {e}")
        return None


def deserialize_model(model_str: str, model_type: str, dataset_name: str) -> Union[nn.Module, None]: # Use Union
    """Deserializes a Base64 string and loads it into a new model instance."""
    is_initial_state = (
        not model_str or
        model_str.strip() == INITIAL_MODEL_STR
    )

    model = None
    try:
        # Always create a base model instance first
        model = _create_new_model_instance(model_type, dataset_name)
        if model is None: return None # Creation failed

        if is_initial_state:
            print(f"\n>> No valid model data/Initial state. Returning new {model_type.upper()} for {dataset_name.upper()}.")
            return model
        else:
            # Attempt to load the state dict
            model_bytes = base64.b64decode(model_str.encode('utf-8'))
            buffer = io.BytesIO(model_bytes)
            buffer.seek(0)
            model.load_state_dict(torch.load(buffer))
            print(f">> Model state ({model_type.upper()} for {dataset_name.upper()}) loaded successfully.")
            return model

    except Exception as e:
        print(f"\n>> [ERROR] Deserializing model state failed: {e}")
        print(f">> Model Str (first 100): {model_str[:100]}...")
        print(f">> Returning NEW {model_type.upper()} for {dataset_name.upper()} as fallback.")
        # Re-create a clean instance as fallback
        return _create_new_model_instance(model_type, dataset_name)