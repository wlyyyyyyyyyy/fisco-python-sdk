# -*- coding: utf-8 -*-
# Model definitions (MLP, CNN). Python 3.7 compatible.

import torch
import torch.nn as nn

# ========== Model Definitions ==========

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron."""
    def __init__(self, input_dim=28*28, hidden_dim=128, output_dim=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        print(f"Initialized MLP: input={input_dim}, hidden={hidden_dim}, output={output_dim}")

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    """A simple Convolutional Neural Network, structure suitable for CIFAR-10."""
    def __init__(self, input_channels=3, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Calculate flattened size dynamically
        fc_input_features = self._get_conv_output_size(input_channels)

        self.fc = nn.Linear(fc_input_features, num_classes)
        print(f"Initialized CNN: input_channels={input_channels}, num_classes={num_classes}, fc_in_features={fc_input_features}")

    def _get_conv_output_size(self, input_channels):
        # Helper method to calculate flattened size
        with torch.no_grad():
             # Use a plausible input size like 32x32
             dummy_input = torch.zeros(1, input_channels, 32, 32)
             dummy_output = self._forward_features(dummy_input)
             return dummy_output.numel() # Total number of elements

    def _forward_features(self, x):
        # Helper method for convolutional part only
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x