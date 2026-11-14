"""
==================================================
AUREX — TEXT GENERATION
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Generates text autoregressively using the transformer.

Status: Initial Development
==================================================
"""

import torch
import torch.nn.functional as F


class AurexGenerator:

    def __init__(self, model, tokenizer, max_seq_length=256):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=1.0):

        if temperature <= 0:
            raise ValueError("Temperature must be greater than zero.")

        self.model.eval()

        input_ids = self.tokenizer.encode(prompt)

        if not input_ids:
            raise ValueError("Prompt cannot be empty.")

        device = next(self.model.parameters()).device
        input_ids = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=device
        )

        for _ in range(max_new_tokens):

            context = input_ids[:, -self.max_seq_length:]

            logits = self.model(context)
            next_token_logits = logits[:, -1, :] / temperature

            probabilities = F.softmax(next_token_logits, dim=-1)

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

        return self.tokenizer.decode(input_ids[0].tolist())
