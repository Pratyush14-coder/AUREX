"""
==================================================
AUREX — INFERENCE ENGINE
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Manages model loading and text generation.

Status: Initial Development
==================================================
"""

class AurexInference:

    def __init__(self, model=None):
        self.model = model

    def load_model(self, model):
        self.model = model

    def generate(self, inputs):
        if self.model is None:
            raise RuntimeError("AUREX model is not loaded.")

        return self.model(inputs)
