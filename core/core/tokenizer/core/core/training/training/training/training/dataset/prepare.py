"""
==================================================
AUREX — DATASET PREPARATION
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Cleans, validates and prepares text datasets
for AUREX model training.

Status: Initial Development
==================================================
"""


class AurexDatasetPreparer:

    def clean_text(self, text):
        if not isinstance(text, str):
            raise TypeError("Dataset entries must be text.")

        return " ".join(text.split())

    def prepare(self, dataset):
        return [
            self.clean_text(text)
            for text in dataset
            if isinstance(text, str) and text.strip()
        ]
