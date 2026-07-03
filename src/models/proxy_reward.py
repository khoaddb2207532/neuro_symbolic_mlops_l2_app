"""Mạng proxy học để thay thế OneVsRestClassifier(LogisticRegression) đắt đỏ
trong vòng lặp reward của GFlowNet — cho phép reward được tính trên GPU, nhanh
hơn nhiều bậc so với gọi sklearn ở mỗi bước sample."""
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ProxyRewardNet(nn.Module):
    def __init__(self, n_rules: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_rules, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.net(mask.float()).squeeze(-1)

    def pretrain(
        self,
        true_reward_fn: Callable,
        n_rules: int,
        device: torch.device,
        n_samples: int = 2000,
        epochs: int = 30,
        lr: float = 1e-3,
    ) -> None:
        logger.info("Pretraining ProxyRewardNet với %d samples...", n_samples)
        X, y = [], []
        for _ in range(n_samples):
            k = np.random.randint(1, min(n_rules, 50) + 1)
            idx = np.random.choice(n_rules, k, replace=False)
            m = torch.zeros(n_rules)
            m[idx] = 1.0
            X.append(m)
            y.append(float(true_reward_fn(m.to(device))))
        X = torch.stack(X).to(device)
        y = torch.tensor(y, device=device, dtype=torch.float32)
        self.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for ep in range(epochs):
            perm = torch.randperm(len(X), device=device)
            loss_sum = 0.0
            for i in range(0, len(X), 256):
                b = perm[i : i + 256]
                l = F.mse_loss(self(X[b]), y[b])
                opt.zero_grad()
                l.backward()
                opt.step()
                loss_sum += l.item()
            if (ep + 1) % 10 == 0:
                logger.info("  Epoch %d/%d loss=%.4f", ep + 1, epochs, loss_sum)
        logger.info("Pretrain ProxyRewardNet xong.")
