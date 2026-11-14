"""
==================================================
AUREX — TOKENIZER
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Converts text into numerical tokens and back.

Status: Initial Development
==================================================
"""

class AurexTokenizer:

    def __init__(self):
        self.vocabulary = {}
        self.reverse_vocabulary = {}

    def train(self, text):
        characters = sorted(set(text))

        self.vocabulary = {
            char: index
            for index, char in enumerate(characters)
        }

        self.reverse_vocabulary = {
            index: char
            for char, index in self.vocabulary.items()
        }

    def encode(self, text):
        return [
            self.vocabulary[char]
            for char in text
            if char in self.vocabulary
        ]

    def decode(self, tokens):
        return "".join(
            self.reverse_vocabulary[token]
            for token in tokens
            if token in self.reverse_vocabulary
        )
