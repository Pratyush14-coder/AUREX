"""
==================================================
AUREX — TOKENIZER TESTS
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Tests tokenizer encoding, decoding and unknown tokens.

Status: Initial Development
==================================================
"""

from tokenizer.tokenizer import AurexTokenizer


def test_tokenizer():

    tokenizer = AurexTokenizer()

    tokenizer.build_vocab("AUREX is an independent AI project.")

    text = "AUREX"

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert len(encoded) == len(text)
    assert decoded == text

    print("Tokenizer test passed!")


if __name__ == "__main__":
    test_tokenizer()
