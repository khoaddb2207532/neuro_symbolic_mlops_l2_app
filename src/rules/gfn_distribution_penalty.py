"""Differentiable CNN penalty backed by frozen GFlowNet leaf probabilities."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from src.rules.rule_types import RuleSet


class GFlowNetDistributionPenalty(nn.Module):
    """Hard RF routing in the forward pass, soft split gradients in backward."""

    requires_labels = True

    def __init__(self, rules: RuleSet, leaf_probs: Mapping, num_classes: int,
                 alpha: float, beta: float, temperature: float = 10.0,
                 epsilon: float = 1e-8):
        super().__init__()
        self.rules, self.alpha, self.beta = rules, float(alpha), float(beta)
        self.penalty_weight = 1.0
        self.epsilon = epsilon
        self.register_buffer("_temperature", torch.tensor(float(temperature)))
        max_conditions = max((len(rule.conditions) for rule in rules.rules), default=1)
        n_rules = len(rules.rules)
        feature_idx = torch.zeros(n_rules, max_conditions, dtype=torch.long)
        thresholds = torch.zeros(n_rules, max_conditions)
        is_greater = torch.zeros(n_rules, max_conditions, dtype=torch.bool)
        valid = torch.zeros(n_rules, max_conditions, dtype=torch.bool)
        good = torch.zeros(n_rules, num_classes)
        bad = torch.zeros(n_rules, num_classes)
        for leaf_id, rule in enumerate(rules.rules):
            for position, condition in enumerate(rule.conditions):
                feature_idx[leaf_id, position] = condition.feature_index
                thresholds[leaf_id, position] = condition.threshold
                is_greater[leaf_id, position] = condition.operator == ">"
                valid[leaf_id, position] = True
            for class_id, table in leaf_probs.get("good", {}).items():
                good[leaf_id, int(class_id)] = float(table.get(str(leaf_id), 0.0))
            for class_id, table in leaf_probs.get("bad", {}).items():
                bad[leaf_id, int(class_id)] = float(table.get(str(leaf_id), 0.0))
        self.register_buffer("feature_idx", feature_idx)
        self.register_buffer("thresholds", thresholds)
        self.register_buffer("is_greater", is_greater)
        self.register_buffer("valid_conditions", valid)
        self.register_buffer("good_probs", good)
        self.register_buffer("bad_probs", bad)
        self._last = {}

    def memberships(self, features: torch.Tensor) -> torch.Tensor:
        selected = features[:, self.feature_idx]
        hard_conditions = torch.where(self.is_greater, selected > self.thresholds,
                                      selected <= self.thresholds)
        hard = torch.where(self.valid_conditions, hard_conditions,
                           torch.ones_like(hard_conditions)).all(dim=-1).float()
        soft_conditions = torch.where(
            self.is_greater,
            torch.sigmoid(self._temperature * (selected - self.thresholds)),
            torch.sigmoid(self._temperature * (self.thresholds - selected)),
        )
        soft = torch.where(self.valid_conditions, soft_conditions,
                           torch.ones_like(soft_conditions)).prod(dim=-1)
        return soft + (hard - soft).detach()

    def forward(self, features: torch.Tensor, logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        membership = self.memberships(features)
        columns = labels.long().unsqueeze(0).expand(len(self.rules.rules), -1)
        p_good = self.good_probs.gather(1, columns).transpose(0, 1)
        p_bad = self.bad_probs.gather(1, columns).transpose(0, 1)
        good_mass = (membership * p_good).sum(dim=1)
        bad_mass = (membership * p_bad).sum(dim=1)
        has_good = p_good.sum(dim=1) > 0
        has_bad = p_bad.sum(dim=1) > 0
        good_term = -self.alpha * torch.log(good_mass[has_good] + self.epsilon).mean() \
            if has_good.any() else features.new_zeros(())
        bad_term = self.beta * torch.log(bad_mass[has_bad] + self.epsilon).mean() \
            if has_bad.any() else features.new_zeros(())
        self._last = {"good_mass": float(good_mass.mean().detach()),
                      "bad_mass": float(bad_mass.mean().detach())}
        return self.penalty_weight * (good_term + bad_term)

    def hard_bad_leaf_ids(self, features: torch.Tensor) -> set[int]:
        with torch.no_grad():
            hard = self.memberships(features).detach() > 0.5
            bad_rules = self.bad_probs.sum(dim=1) > 0
            return set(torch.where((hard & bad_rules.unsqueeze(0)).any(dim=0))[0].cpu().tolist())

    def hard_bad_leaf_keys(self, features: torch.Tensor, labels: torch.Tensor) -> set[tuple[int, int]]:
        """Active (class, leaf) pairs, matching the definition of total |B_c|."""
        with torch.no_grad():
            hard = self.memberships(features).detach() > 0.5
            active = set()
            for row, class_id in enumerate(labels.long().tolist()):
                valid_bad = self.bad_probs[:, class_id] > 0
                active.update((class_id, int(i)) for i in torch.where(hard[row] & valid_bad)[0].cpu())
            return active

    def last_coverage_stats(self) -> dict:
        return {"n_rules_total": len(self.rules.rules),
                "n_rules_active_this_batch": 0,
                "mean_rule_sat": self._last.get("good_mass", 0.0)}
