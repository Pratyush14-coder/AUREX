"""
==================================================
AUREX — TRAINING METRICS
Created by: Pratyush Rao
Project: XO AI Ecosystem
Version: 0.1.0
==================================================

Description:
Tracks and summarizes model training loss.

Status: Initial Development
==================================================
"""


class AurexMetrics:

    def __init__(self):
        self.loss_history = []

    def record_loss(self, loss):
        self.loss_history.append(float(loss))

    def average_loss(self):
        if not self.loss_history:
            return 0.0

        return sum(self.loss_history) / len(self.loss_history)

    def latest_loss(self):
        if not self.loss_history:
            return None

        return self.loss_history[-1]

    def reset(self):
        self.loss_history.clear()
