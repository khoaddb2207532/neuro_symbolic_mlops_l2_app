import pytest
import numpy  # Load NumPy before Torch to avoid the current Windows MKL import-order crash.
import torch

from src.gflownet.leaf_path import LeafPathGraph, train_leaf_path_gflownet
from src.gflownet.validation import validate_leaf_probability_coverage
from src.rules.rule_types import Condition, Rule


def _rules():
    return {
        10: Rule([Condition(0, "<=", 0.5)], 0, tree_id=0, leaf_node_id=1),
        11: Rule([Condition(0, ">", 0.5)], 0, tree_id=0, leaf_node_id=2),
    }


def test_graph_actions_are_only_real_rf_path_edges():
    graph = LeafPathGraph(_rules())
    assert set(graph.leaf_to_node) == {10, 11}
    assert len(graph.nodes[0].children) == 1  # only tree 0
    tree_root = next(iter(graph.nodes[0].children.values()))
    assert len(graph.nodes[tree_root].children) == 2
    assert all(graph.nodes[node].leaf_id in {10, 11} for node in graph.leaf_to_node.values())


def test_tb_policy_normalizes_over_filtered_terminals():
    model, summary = train_leaf_path_gflownet(
        _rules(), {10: 0.8, 11: 0.2}, hidden_dim=16, n_layers=2,
        learning_rate=3e-3, steps=300, exploration=0.1, reward_beta=1.0,
        kl_patience=20, kl_tolerance=1e-5, seed=3, device="cpu",
    )
    probabilities = model.terminal_probabilities()
    assert set(probabilities) == {10, 11}
    assert float(sum(probabilities.values()).detach()) == pytest.approx(1.0, abs=1e-6)
    assert probabilities[10] > probabilities[11]
    assert summary["steps"] <= 300
    assert summary["torchgfn"] is True


def test_probability_table_must_cover_exact_filtered_universe():
    probabilities = {"good": {"0": {"10": 0.4, "11": 0.6}},
                     "bad": {"0": {"12": 1.0}}}
    validate_leaf_probability_coverage(probabilities, {"0": [10, 11]}, {"0": [12]})
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        validate_leaf_probability_coverage(probabilities, {"0": [10, 13]}, {"0": [12]})
