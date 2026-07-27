"""Optimizer builder cho CNN với learning rate cố định."""
from typing import Dict

import torch.nn as nn
import torch.optim as optim


def build_optimizer(model: nn.Module, config: Dict):
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)

    if hasattr(model, "trainable_param_groups") and "lr_backbone" in config and "lr_head" in config:
        # Chiến lược Differential LR: backbone (nếu đang mở) học chậm hơn nhiều
        # so với head, để không phá vỡ đặc trưng pretrained.
        param_groups = model.trainable_param_groups(
            lr_head=config["lr_head"], lr_backbone=config["lr_backbone"]
        )
        for g in param_groups:
            g["weight_decay"] = weight_decay
        return optim.Adam(param_groups)

    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
