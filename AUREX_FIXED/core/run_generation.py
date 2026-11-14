"""
==================================================
AUREX — GENERATION LAUNCHER
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.2.0
==================================================

Description:
Loads the saved tokenizer and trained model checkpoint
to generate text.

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

    tokenizer = AurexTokenizer()
    tokenizer.load("aurex_tokenizer.json")

    model = AurexTransformer(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_length=config.max_seq_length
    ).to(device)

    checkpoint = torch.load(
        config.checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    generator = AurexGenerator(
        model,
        tokenizer,
        max_seq_length=config.max_seq_length
    )

    prompt = input("You: ")

    result = generator.generate(
        prompt,
        max_new_tokens=60,
        temperature=0.8
    )

    print("\nAUREX:", result)


if __name__ == "__main__":
    main()
