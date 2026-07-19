import math

import pytest
import torch

from src.rules.gfn_distribution_penalty import GFlowNetDistributionPenalty
from src.rules.rule_types import Condition, Rule, RuleSet


def test_distribution_penalty_matches_formula_and_has_feature_gradient():
    rules = RuleSet([
        Rule([Condition(0, "<=", 0.0)], 0),
        Rule([Condition(0, ">", 0.0)], 0),
    ])
    probabilities = {"good": {"0": {"0": 1.0}},
                     "bad": {"0": {"1": 1.0}}}
    epsilon = 1e-6
    penalty = GFlowNetDistributionPenalty(
        rules, probabilities, num_classes=1, alpha=2.0, beta=1.0,
        temperature=5.0, epsilon=epsilon,
    )
    features = torch.tensor([[-1.0], [1.0]], requires_grad=True)
    logits = torch.zeros(2, 1)
    labels = torch.zeros(2, dtype=torch.long)
    value = penalty(features, logits, labels)
    expected = -2.0 * math.log(epsilon) / 2 + math.log(epsilon) / 2
    assert float(value.detach()) == pytest.approx(expected, rel=1e-4)
    value.backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_hard_forward_membership_matches_rf_conditions():
    rules = RuleSet([Rule([Condition(0, "<=", 0.0)], 0),
                     Rule([Condition(0, ">", 0.0)], 0)])
    penalty = GFlowNetDistributionPenalty(rules, {}, 1, 0.1, 0.1)
    membership = penalty.memberships(torch.tensor([[-1.0], [1.0]]))
    assert torch.equal(membership.detach(), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
