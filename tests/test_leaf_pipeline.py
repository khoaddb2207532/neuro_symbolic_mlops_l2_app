import numpy as np

from src.rules.leaf_analysis import build_leaf_stats, classify_leaves, wilson_lower_bound
from src.rules.rule_types import Condition, Rule, RuleSet
from src.training.drift import drift_decision


def test_wilson_lower_bound_is_conservative():
    assert 0 < wilson_lower_bound(9, 10) < 0.9
    assert wilson_lower_bound(0, 0) == 0


def test_leaf_stats_use_cnn_agreement_for_fidelity():
    rules = RuleSet([Rule([Condition(0, "<=", 0.5)], target_class=1)])
    x = np.array([[0.1], [0.2], [0.9]])
    stats = build_leaf_stats(rules, x, np.array([1, 0, 0]), np.array([1, 1, 0]))
    assert stats.iloc[0].coverage == 2
    assert stats.iloc[0].fidelity == 1.0
    assert stats.iloc[0].precision == 0.5


def test_drift_trigger_is_relative():
    assert drift_decision(0.79, 0.90, 0.10)["refit_required"] is True
    assert drift_decision(0.82, 0.90, 0.10)["refit_required"] is False
