"""Optimizer/scheduler builder — áp dụng Differential Learning Rate theo stage
đóng băng hiện tại của model (xem src/models/cnn.py::CNNBaseline.set_freeze_stage)."""
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
