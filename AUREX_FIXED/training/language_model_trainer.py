"""
==================================================
AUREX — LANGUAGE MODEL TRAINER
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Trains the AUREX transformer using next-token prediction.

Status: Initial Development
==================================================
"""

import torch
import torch.nn as nn

from core.transformer import AurexTransformer
from dataset.dataloader import create_dataloader
from core.model_config import AurexModelConfig
from training.checkpoint import AurexCheckpoint
from training.metrics import AurexMetrics


class AurexLanguageModelTrainer:

    def __init__(self, model, learning_rate=0.0003):

        self.model = model

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.CrossEntropyLoss()
        self.metrics = AurexMetrics()

        self.device = next(model.parameters()).device

    def train(self, token_ids, epochs=3, batch_size=16):

        loader = create_dataloader(
            token_ids,
            batch_size=batch_size,
            sequence_length=64
        )

        self.model.train()

        for epoch in range(epochs):

            total_loss = 0.0
            batches = 0

            for inputs, targets in loader:

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.loss_function(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1)
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                self.optimizer.step()

                total_loss += loss.item()
                batches += 1

            if batches == 0:
                raise ValueError(
                    "Not enough tokens to create a training batch."
                )

            average_loss = total_loss / batches
            self.metrics.record_loss(average_loss)

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"| Loss: {average_loss:.4f}"
            )

            AurexCheckpoint.save(
                self.model,
                self.optimizer,
                epoch + 1
            )

        return self.metrics.loss_history
