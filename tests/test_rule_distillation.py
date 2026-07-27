import torch
import torch.nn as nn

from src.rules.distillation import RuleDistillationPenalty
from src.rules.rule_types import Condition, Rule, RuleSet


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)

    def forward(self, images):
        features = self.projection(images)
        return features, features


def test_rule_distillation_backpropagates_only_to_student():
    teacher = TinyTeacher()
    rules = RuleSet(
        [Rule([Condition(0, ">", 0.0)], target_class=1, confidence=0.9)]
    )
    penalty = RuleDistillationPenalty(
        teacher, rules, num_classes=2, penalty_weight=0.2
    )
    images = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    student_logits = torch.randn(2, 2, requires_grad=True)

    loss = penalty(images, student_logits)
    loss.backward()

    assert loss.item() >= 0.0
    assert student_logits.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_empty_rules_return_zero_loss():
    logits = torch.randn(2, 3, requires_grad=True)
    penalty = RuleDistillationPenalty(
        TinyTeacher(), RuleSet([]), num_classes=3
    )
    loss = penalty(torch.randn(2, 2), logits)
    assert loss.item() == 0.0
