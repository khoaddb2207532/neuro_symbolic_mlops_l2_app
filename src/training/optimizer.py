"""Optimizer cho huấn luyện end-to-end với một learning rate cố định."""
from typing import Dict

import torch.nn as nn
import torch.optim as optim


def build_optimizer(model: nn.Module, config: Dict):
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)

    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
