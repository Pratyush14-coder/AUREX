"""
==================================================
AUREX — MODEL CONFIGURATION
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Stores the architecture and model configuration.

Status: Initial Development
==================================================
"""


class AurexModelConfig:

    def __init__(self):
        self.vocab_size = 256
        self.embedding_dim = 128
        self.num_heads = 4
        self.num_layers = 2
        self.max_seq_length = 256
        self.dropout = 0.1

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_seq_length": self.max_seq_length,
            "dropout": self.dropout
        }
