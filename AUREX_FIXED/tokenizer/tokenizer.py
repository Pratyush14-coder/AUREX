"""
==================================================
AUREX — TOKENIZER
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.2.0
==================================================

Description:
Converts text into numerical tokens and back.
Supports unknown characters and vocabulary persistence.

Status: Initial Development
==================================================
"""

import json


class AurexTokenizer:

    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.unknown_token = 0

    def build_vocab(self, text):

        characters = sorted(set(text))

        self.vocab = {
            "<UNK>": self.unknown_token
        }

        for char in characters:
            if char != "<":
                self.vocab[char] = len(self.vocab)

        self.reverse_vocab = {
            index: token
            for token, index in self.vocab.items()
        }

    def train(self, text):
        self.build_vocab(text)

    def encode(self, text):

        if not self.vocab:
            raise ValueError("Tokenizer vocabulary is empty.")

        return [
            self.vocab.get(char, self.unknown_token)
            for char in text
        ]

    def decode(self, tokens):

        return "".join(
            self.reverse_vocab.get(token, "")
            for token in tokens
            if token != self.unknown_token
        )

    def save(self, path="aurex_tokenizer.json"):

        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False)

    def load(self, path="aurex_tokenizer.json"):

        with open(path, "r", encoding="utf-8") as file:
            self.vocab = json.load(file)

        self.vocab = {
            token: int(index)
            for token, index in self.vocab.items()
        }

        self.reverse_vocab = {
            index: token
            for token, index in self.vocab.items()
        }
