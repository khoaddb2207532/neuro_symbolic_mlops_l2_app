import torch

from src.rules.penalty import BinaryTransformer
from src.rules.rule_types import Condition, Rule, RuleSet


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
