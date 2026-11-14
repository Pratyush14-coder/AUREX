"""
==================================================
AUREX — DATASET VALIDATOR
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Validates text datasets before model training.

Status: Initial Development
==================================================
"""

class AurexDatasetValidator:

    def validate(self, text):

        if not isinstance(text, str):
            raise TypeError("Dataset must be text.")

        if not text.strip():
            raise ValueError("Dataset cannot be empty.")

        if len(text) < 100:
            raise ValueError("Dataset is too small for training.")

        return {
            "valid": True,
            "characters": len(text),
            "words": len(text.split())
        }


if __name__ == "__main__":

    sample = "AUREX dataset validation sample. " * 10

    validator = AurexDatasetValidator()

    print(validator.validate(sample))
