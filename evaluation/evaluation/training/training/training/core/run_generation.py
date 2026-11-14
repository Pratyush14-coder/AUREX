"""
==================================================
AUREX — GENERATION LAUNCHER
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Loads a trained AUREX model and generates text.

Status: Initial Development
==================================================
"""

import torch

from tokenizer.tokenizer import AurexTokenizer
from core.transformer import AurexTransformer
from core.generation import AurexGenerator
from training.language_config import AurexLanguageConfig


def main():

    config = AurexLanguageConfig()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # This must match the tokenizer used during training.
    tokenizer = AurexTokenizer()

    training_text = (
        "AUREX is an independent artificial intelligence project. "
        "It learns language patterns through next token prediction. "
    ) * 100

    tokenizer.build_vocab(training_text)

    model = AurexTransformer(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_length=config.max_seq_length
    ).to(device)

    checkpoint = torch.load(
        config.checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    generator = AurexGenerator(
        model,
        tokenizer,
        max_seq_length=config.max_seq_length
    )

    result = generator.generate(
        "AUREX is",
        max_new_tokens=40
    )

    print("\nAUREX OUTPUT:")
    print(result)


if __name__ == "__main__":
    main()
