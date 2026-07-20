"""Optimizer/scheduler cho huấn luyện end-to-end với một learning rate."""
from typing import Dict

import torch.nn as nn
import torch.optim as optim


def build_optimizer(model: nn.Module, config: Dict):
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)

    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, config: Dict):
    if config.get("use_scheduler", False):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.get("scheduler_factor", 0.1),
            patience=config.get("scheduler_patience", 3),
            threshold=config.get("scheduler_threshold", 1e-4),
            cooldown=config.get("scheduler_cooldown", 0),
            min_lr=config.get("scheduler_min_lr", 1e-6),
        )
    return None
