import torch

from src.rules.penalty import BinaryTransformer
from src.rules.rule_types import Condition, Rule, RuleSet
from src.gflownet.reward import RuleSetReward


def test_ruleset_len_and_iter():
    rules = [Rule([Condition(0, "<=", 0.5)], target_class=1)]
    rs = RuleSet(rules=rules)
    assert len(rs) == 1
    assert list(rs) == rules


def test_binary_transformer_shapes():
    features = torch.rand(8, 4)
    rule = Rule([Condition(0, "<=", 0.5), Condition(1, ">", 0.2)], target_class=0)
    rs = RuleSet(rules=[rule])
    out = BinaryTransformer().transform(features, rs)
    assert out.shape == (8, 1)
    assert torch.all((out >= 0) & (out <= 1))


def test_binary_transformer_empty_ruleset():
    features = torch.rand(5, 3)
    out = BinaryTransformer().transform(features, RuleSet(rules=[]))
    assert out.shape == (5, 0)
def test_ruleset_reward_uses_macro_correct_and_wrong_coverage():
    cover = torch.tensor(
        [
            [1, 0],  # target 0, đúng trên sample 0
            [0, 1],  # target 1, đúng trên sample 1
            [0, 1],  # target 0, sai trên sample 1
        ],
        dtype=torch.bool,
    )
    correct = torch.tensor(
        [
            [1, 0],
            [0, 1],
            [0, 0],
        ],
        dtype=torch.bool,
    )
    reward = RuleSetReward(
        cover=cover,
        correct=correct,
        rule_len=torch.ones(3),
        max_rules=3,
        targets=torch.tensor([0, 1, 0]),
        labels=torch.tensor([0, 1]),
        confidences=torch.ones(3),
        w_acc=1.0,
        w_cov=0.5,
        w_wrong=0.75,
        w_conflict=0.1,
        beta=1.0,
    )

    good = reward.components(torch.tensor([[1.0, 1.0, 0.0]]))
    assert good["macro_accuracy"].item() == 1.0
    assert good["correct_coverage"].item() == 1.0
    assert good["wrong_coverage"].item() == 0.0
    assert good["conflict_ratio"].item() == 0.0

    conflicting_wrong = reward.components(torch.tensor([[0.0, 1.0, 1.0]]))
    assert conflicting_wrong["correct_coverage"].item() == 0.0
    assert conflicting_wrong["wrong_coverage"].item() == 0.5
    assert conflicting_wrong["conflict_ratio"].item() > 0.0
