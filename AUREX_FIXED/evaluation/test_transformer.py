"""
==================================================
AUREX — TRANSFORMER TESTS
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Tests transformer output dimensions and forward execution.

Status: Initial Development
==================================================
"""

import torch
from core.transformer import AurexTransformer


def test_transformer():

    vocab_size = 256
    model = AurexTransformer(vocab_size=vocab_size)

    input_ids = torch.randint(
        0,
        vocab_size,
        (2, 32)
    )

    output = model(input_ids)

    assert output.shape == (2, 32, vocab_size)
    assert torch.isfinite(output).all()

    print("Transformer test passed!")


if __name__ == "__main__":
    test_transformer()
