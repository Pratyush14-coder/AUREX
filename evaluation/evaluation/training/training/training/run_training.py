"""
==================================================
AUREX — TRAINING LAUNCHER
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Initializes the tokenizer, transformer and training pipeline.

Status: Initial Development
==================================================
"""

import torch

from tokenizer.tokenizer import AurexTokenizer
from core.transformer import AurexTransformer
from training.language_model_trainer import AurexLanguageModelTrainer
from training.language_config import AurexLanguageConfig


def main():

    config = AurexLanguageConfig()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"AUREX training device: {device}")

    # Small starter text; replace with an approved dataset later.
    training_text = (
        "AUREX is an independent artificial intelligence project. "
        "It learns language patterns through next token prediction. "
    ) * 100

    tokenizer = AurexTokenizer()
    tokenizer.build_vocab(training_text)

    token_ids = tokenizer.encode(training_text)

    model = AurexTransformer(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_length=config.max_seq_length
    ).to(device)

    trainer = AurexLanguageModelTrainer(
        model,
        learning_rate=config.learning_rate
    )

    trainer.train(
        token_ids,
        epochs=config.epochs,
        batch_size=config.batch_size
    )

    print("AUREX training run completed.")


if __name__ == "__main__":
    main()
