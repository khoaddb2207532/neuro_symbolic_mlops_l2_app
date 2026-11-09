import json

import pytest
import torch

from src.rules.bayesian_penalty import BayesianRuleMarginalization
from src.rules.rule_types import Condition, Rule


class _TerminatingStates:
    def __init__(self, tensor):
        self.tensor = tensor


class _Trajectories:
    def __init__(self, tensor):
        self.terminating_states = _TerminatingStates(tensor)


class _FrozenSampler(torch.nn.Module):
    def __init__(self, masks):
        super().__init__()
        self.masks = masks

    def sample_trajectories(self, env, n, save_logprobs=False):
        assert n == self.masks.size(0)
        return _Trajectories(self.masks)


class _Env:
    n_rules = 2


def test_rule_contributions_reconcile_with_returned_penalty(tmp_path):
    rules = [
        Rule([Condition(0, "<=", 0.5)], target_class=0, confidence=1.0),
        Rule([Condition(0, ">", 0.5)], target_class=1, confidence=0.8),
    ]
    masks = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
    module = BayesianRuleMarginalization(
        valid_rules=rules,
        gflownet=_FrozenSampler(masks),
        env=_Env(),
        K=2,
        penalty_weight=0.2,
        use_confidence=True,
        num_classes=2,
    )

    features = torch.tensor([[0.0], [1.0]])
    logits = torch.tensor([[0.2, -0.1], [-0.2, 0.4]], requires_grad=True)
    penalty = module(features, logits)

    assert module._contribution_sum.sum().item() == pytest.approx(
        penalty.detach().item(), rel=1e-6
    )
    assert module._normalized_weight_sum.tolist() == pytest.approx([0.75, 0.25])
    assert module._contribution_step_count.item() == 1

    summary = module.save_sampling_summary(str(tmp_path))
    assert summary["contribution_steps"] == 1
    assert summary["total_cumulative_penalty_contribution"] == pytest.approx(
        penalty.detach().item(), rel=1e-6
    )
    assert summary["top_impact_rules"][0]["rank"] == 1
    assert json.loads(
        (tmp_path / "gflownet_sampling_summary.json").read_text(encoding="utf-8")
    )["top_impact_rules"]
