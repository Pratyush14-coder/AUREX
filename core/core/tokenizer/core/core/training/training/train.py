"""
==================================================
AUREX — TRAINING ENGINE
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Controls the model training process and
manages training progress.

Status: Initial Development
==================================================
"""

import torch


class AurexTrainer:

    def __init__(self, model, optimizer, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

    def train_step(self, inputs, targets):

        self.model.train()

        self.optimizer.zero_grad()

        predictions = self.model(inputs)

        loss = self.loss_function(predictions, targets)

        loss.backward()

        self.optimizer.step()

        return loss.item()
