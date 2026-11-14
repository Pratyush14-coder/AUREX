"""
==================================================
AUREX — TRAINING LAUNCHER
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.2.0
==================================================

Description:
Initializes, trains and saves the AUREX language model.

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

    training_text = (
        "AUREX is an independent artificial intelligence project. "
        "It learns language patterns through next token prediction. "
    ) * 100

    tokenizer = AurexTokenizer()
    tokenizer.build_vocab(training_text)

    # Save tokenizer for future generation.
    tokenizer.save("aurex_tokenizer.json")

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

    print("AUREX training completed.")
    print("Tokenizer and model checkpoint saved.")


if __name__ == "__main__":
    main()
