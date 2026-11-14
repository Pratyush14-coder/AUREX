"""
==================================================
AUREX — TRANSFORMER ARCHITECTURE
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.2.0
==================================================

Description:
Decoder-style transformer for autoregressive
next-token language modelling.

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
        max_seq_length=256,
        dropout=0.1
    ):
        super().__init__()

        self.max_seq_length = max_seq_length

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
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.output_layer = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False
        )

    def forward(self, input_ids):

        batch_size, seq_length = input_ids.shape

        if seq_length > self.max_seq_length:
            raise ValueError("Sequence exceeds maximum length.")

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
            x = layer(
                x,
                src_mask=causal_mask,
                is_causal=True
            )

        x = self.final_norm(x)

        return self.output_layer(x)
