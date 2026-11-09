"""Distill tri thức từ các luật sang một CNN student.

Teacher CNN chỉ dùng để chiếu ảnh vào đúng không gian đặc trưng nơi các luật
được khai phá. Student nhận gradient từ CE và KL với phân phối lớp do luật bỏ
phiếu; student không kế thừa trọng số của teacher.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rules.rule_types import RuleSet
from src.rules.temperature import geometric_temperature


class RuleDistillationPenalty(nn.Module):
    """KL distillation từ rule votes trên feature của một teacher bị đóng băng."""

    uses_images = True

    def __init__(
        self,
        teacher: nn.Module,
        rule_set: RuleSet,
        num_classes: int,
        penalty_weight: float = 0.1,
        distillation_temperature: float = 2.0,
        initial_temp: float = 2.0,
        final_temp: float = 12.0,
        temp_warmup_epochs: int = 2,
        temp_anneal_epochs: int = 10,
        use_confidence: bool = True,
        eps: float = 1e-8,
        active_threshold: float = 1.0,
    ):
        super().__init__()
        if distillation_temperature <= 0:
            raise ValueError("distillation_temperature phải lớn hơn 0.")
        self.teacher = teacher
        self.rule_set = rule_set
        self.num_classes = num_classes
        self.penalty_weight = penalty_weight
        self.distillation_temperature = distillation_temperature
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.temp_warmup_epochs = temp_warmup_epochs
        self.temp_anneal_epochs = temp_anneal_epochs
        self.use_confidence = use_confidence
        self.eps = eps
        self.active_threshold = active_threshold
        self.register_buffer("_temperature", torch.tensor(float(initial_temp)))
        self._last_rule_sat: Optional[torch.Tensor] = None

        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        """Giữ teacher ở eval mode ngay cả khi penalty được chuyển sang train."""
        super().train(mode)
        self.teacher.eval()
        return self

    def update_temperature(self, epoch: int) -> None:
        value = geometric_temperature(
            epoch,
            self.initial_temp,
            self.final_temp,
            self.temp_warmup_epochs,
            self.temp_anneal_epochs,
        )
        self._temperature.fill_(value)

    def _rule_satisfaction(self, features: torch.Tensor) -> torch.Tensor:
        if len(self.rule_set) == 0:
            return features.new_zeros((features.size(0), 0))

        satisfactions = []
        temperature = self._temperature
        for rule in self.rule_set.rules:
            satisfaction = features.new_ones(features.size(0))
            for condition in rule.conditions:
                column = features[:, condition.feature_index]
                if condition.operator == "<=":
                    condition_sat = torch.sigmoid(
                        temperature * (condition.threshold - column)
                    )
                else:
                    condition_sat = torch.sigmoid(
                        temperature * (column - condition.threshold)
                    )
                satisfaction = satisfaction * condition_sat
            satisfactions.append(satisfaction)
        return torch.stack(satisfactions, dim=1)

    def forward(
        self, images: torch.Tensor, student_logits: torch.Tensor
    ) -> torch.Tensor:
        if len(self.rule_set) == 0:
            self._last_rule_sat = None
            return student_logits.new_zeros(())

        with torch.no_grad():
            teacher_output = self.teacher(images)
            teacher_features = (
                teacher_output[1]
                if isinstance(teacher_output, (tuple, list))
                else teacher_output
            )
            rule_sat = self._rule_satisfaction(teacher_features)
            self._last_rule_sat = rule_sat

            confidences = teacher_features.new_tensor(
                [rule.confidence for rule in self.rule_set.rules]
            )
            if not self.use_confidence:
                confidences.fill_(1.0)
            weighted_votes = rule_sat * confidences.unsqueeze(0)

            targets = torch.tensor(
                [rule.target_class for rule in self.rule_set.rules],
                dtype=torch.long,
                device=images.device,
            )
            evidence = teacher_features.new_zeros(
                (images.size(0), self.num_classes)
            )
            evidence.scatter_add_(
                1, targets.unsqueeze(0).expand(images.size(0), -1), weighted_votes
            )

            coverage = evidence.sum(dim=1)
            rule_probs = (evidence + self.eps) / (
                evidence.sum(dim=1, keepdim=True) + self.eps * self.num_classes
            )
            temperature = self.distillation_temperature
            softened = rule_probs.pow(1.0 / temperature)
            softened = softened / softened.sum(dim=1, keepdim=True)

        student_log_probs = F.log_softmax(
            student_logits / self.distillation_temperature, dim=1
        )
        per_sample_kl = F.kl_div(
            student_log_probs, softened, reduction="none"
        ).sum(dim=1)
        coverage_weight = coverage.clamp(max=1.0)
        denominator = coverage_weight.sum().clamp_min(self.eps)
        distillation_loss = (per_sample_kl * coverage_weight).sum() / denominator
        return (
            self.penalty_weight
            * self.distillation_temperature**2
            * distillation_loss
        )

    def last_coverage_stats(self) -> dict:
        if self._last_rule_sat is None:
            return {
                "n_rules_total": len(self.rule_set),
                "n_rules_active_this_batch": 0,
                "mean_rule_sat": 0.0,
            }
        per_rule_sum = self._last_rule_sat.sum(dim=0)
        return {
            "n_rules_total": len(self.rule_set),
            "n_rules_active_this_batch": int(
                (per_rule_sum >= self.active_threshold).sum().item()
            ),
            "mean_rule_sat": float(self._last_rule_sat.mean().item()),
        }
