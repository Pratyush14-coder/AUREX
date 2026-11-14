"""
==================================================
AUREX — LANGUAGE MODEL CONFIGURATION
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Defines configuration for language model training.

Status: Initial Development
==================================================
"""


class AurexLanguageConfig:

    def __init__(self):

        self.epochs = 3
        self.batch_size = 16
        self.learning_rate = 0.0003
        self.sequence_length = 64
        self.embedding_dim = 128
        self.num_heads = 4
        self.num_layers = 2
        self.max_seq_length = 256
        self.checkpoint_path = "aurex_checkpoint.pt"

    def to_dict(self):

        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "sequence_length": self.sequence_length,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_seq_length": self.max_seq_length,
            "checkpoint_path": self.checkpoint_path
        }
