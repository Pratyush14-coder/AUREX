"""
==================================================
AUREX — MODEL TESTING
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Tests model outputs and validates basic
AUREX functionality.

Status: Initial Development
==================================================
"""


class AurexModelTester:

    def test_tokenizer(self, tokenizer, text):
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        return {
            "original": text,
            "encoded": encoded,
            "decoded": decoded,
            "passed": decoded == text
        }

    def test_response(self, response):
        return {
            "response": response,
            "passed": isinstance(response, str) and bool(response.strip())
        }
