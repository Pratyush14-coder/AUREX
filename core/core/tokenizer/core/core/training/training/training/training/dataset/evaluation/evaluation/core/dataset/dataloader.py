"""
==================================================
AUREX — DATA LOADER
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Converts tokenized text into training sequences.

Status: Initial Development
==================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader


class AurexDataset(Dataset):

    def __init__(self, token_ids, sequence_length=64):
        self.token_ids = token_ids
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.token_ids) - self.sequence_length)

    def __getitem__(self, index):
        x = self.token_ids[index:index + self.sequence_length]
        y = self.token_ids[index + 1:index + self.sequence_length + 1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


def create_dataloader(token_ids, batch_size=16, sequence_length=64):

    dataset = AurexDataset(token_ids, sequence_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
