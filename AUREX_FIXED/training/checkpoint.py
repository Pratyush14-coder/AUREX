"""
==================================================
AUREX — CHECKPOINT MANAGER
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Saves and loads model training checkpoints.

Status: Initial Development
==================================================
"""

import torch
import os


class AurexCheckpoint:

    @staticmethod
    def save(model, optimizer, epoch, path="aurex_checkpoint.pt"):

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, path)

    @staticmethod
    def load(model, optimizer, path):

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        checkpoint = torch.load(
            path,
            map_location="cpu"
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint["epoch"]
