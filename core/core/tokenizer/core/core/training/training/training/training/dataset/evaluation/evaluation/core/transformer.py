
"""
==================================================
AUREX — TRANSFORMER ARCHITECTURE
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Defines the initial decoder-only transformer architecture.

Status: Initial Development
==================================================
"""

import torch
import torch.nn as nn


class AurexTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        max_seq_length=256
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            max_seq_length,
            embedding_dim
        )

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, input_ids):

        batch_size, seq_length = input_ids.shape

        positions = torch.arange(
            seq_length,
            device=input_ids.device
        ).unsqueeze(0)

        x = (
            self.embedding(input_ids)
            + self.position_embedding(positions)
        )

        causal_mask = torch.triu(
            torch.ones(
                seq_length,
                seq_length,
                device=input_ids.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)

        return self.output_layer(x)
