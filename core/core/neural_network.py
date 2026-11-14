"""
AUREX Neural Network
Created by Pratyush Rao
"""

import torch
import torch.nn as nn


class AurexNeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, inputs):
        return self.network(inputs)
