"""Statistics and statistically defensible filtering for extracted RF leaves."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.rules.rule_types import Rule, RuleSet


def rule_mask(rule: Rule, features: np.ndarray) -> np.ndarray:
    mask = np.ones(len(features), dtype=bool)
    for condition in rule.conditions:
        column = features[:, condition.feature_index]
        mask &= column <= condition.threshold if condition.operator == "<=" else column > condition.threshold
    return mask


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (centre - margin) / denominator


def build_leaf_stats(
    rules: RuleSet | Sequence[Rule],
    features: np.ndarray,
    labels: np.ndarray,
    cnn_predictions: np.ndarray,
) -> pd.DataFrame:
    """Measure each leaf on validation data; no CNN forward pass occurs here."""
    rows = []
    iterable = rules.rules if isinstance(rules, RuleSet) else rules
    for leaf_id, rule in enumerate(iterable):
        covered = rule_mask(rule, features)
        coverage = int(covered.sum())
        fidelity_hits = int((cnn_predictions[covered] == rule.target_class).sum())
        precision_hits = int((labels[covered] == rule.target_class).sum())
        rows.append(
            {
                "leaf_id": leaf_id,
                "target_class": int(rule.target_class),
                "coverage": coverage,
                "coverage_ratio": coverage / max(len(features), 1),
                "fidelity_hits": fidelity_hits,
                "fidelity": fidelity_hits / coverage if coverage else 0.0,
                "wilson_fidelity_lower": wilson_lower_bound(fidelity_hits, coverage),
                "precision_hits": precision_hits,
                "precision": precision_hits / coverage if coverage else 0.0,
                "rule_length": len(rule.conditions),
            }
        )
    return pd.DataFrame(rows)


def build_forest_leaf_stats(rf_model, rules: RuleSet, features: np.ndarray,
                            labels: np.ndarray, cnn_predictions: np.ndarray) -> pd.DataFrame:
    """Measure every tree leaf using the complete forest prediction on validation."""
    forest_predictions = rf_model.predict(features)
    applied_nodes = rf_model.apply(features)
    rows = []
    for leaf_id, rule in enumerate(rules.rules):
            tree_id, node_id = rule.tree_id, rule.leaf_node_id
            covered = applied_nodes[:, tree_id] == node_id
            coverage = int(covered.sum())
            fidelity_hits = int((forest_predictions[covered] == cnn_predictions[covered]).sum())
            precision_hits = int((forest_predictions[covered] == labels[covered]).sum())
            if coverage:
                predicted_class = int(np.bincount(forest_predictions[covered].astype(int)).argmax())
            else:
                predicted_class = int(rule.target_class)
            class_mask = cnn_predictions == predicted_class
            class_total = int(class_mask.sum())
            class_hits = int((labels[class_mask] == predicted_class).sum())
            rows.append({
                "leaf_id": leaf_id, "tree_id": tree_id, "node_id": int(node_id),
                "fidelity": fidelity_hits / coverage if coverage else 0.0,
                "precision": precision_hits / coverage if coverage else 0.0,
                "coverage": coverage, "coverage_ratio": coverage / len(features),
                "fidelity_hits": fidelity_hits,
                "wilson_fidelity_lower": wilson_lower_bound(fidelity_hits, coverage),
                "precision_hits": precision_hits,
                "class_precision": class_hits / class_total if class_total else 0.0,
                "class_precision_hits": class_hits,
                "class_prediction_count": class_total,
                "target_class": predicted_class,
                "rule_length": len(rule.conditions),
            })
    return pd.DataFrame(rows).sort_values("leaf_id").reset_index(drop=True)


def _two_proportion_pvalue(successes: int, total: int, base_successes: int, base_total: int) -> float:
    if total <= 0 or base_total <= 0:
        return 1.0
    pooled = (successes + base_successes) / (total + base_total)
    se = math.sqrt(max(pooled * (1.0 - pooled) * (1.0 / total + 1.0 / base_total), 0.0))
    if se == 0.0:
        return 1.0
    z = abs(successes / total - base_successes / base_total) / se
    return math.erfc(z / math.sqrt(2.0))


def classify_leaves(
    stats: pd.DataFrame,
    labels: np.ndarray,
    fidelity_threshold: float,
    min_coverage: int,
    significance: float = 0.05,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """Split faithful leaves into good/bad groups relative to class prevalence."""
    good: Dict[str, List[int]] = {}
    bad: Dict[str, List[int]] = {}
    for row in stats.to_dict("records"):
        if row["wilson_fidelity_lower"] <= fidelity_threshold or row["coverage"] <= min_coverage:
            continue
        class_id = int(row["target_class"])
        base_hits = int(row.get("class_precision_hits", (labels == class_id).sum()))
        base_total = int(row.get("class_prediction_count", len(labels)))
        pvalue = _two_proportion_pvalue(
            int(row["precision_hits"]), int(row["coverage"]), base_hits, base_total
        )
        if pvalue >= significance:
            continue
        bucket = good if row["precision"] > base_hits / max(base_total, 1) else bad
        bucket.setdefault(str(class_id), []).append(int(row["leaf_id"]))
    return good, bad


def save_leaf_groups(good: Dict[str, List[int]], bad: Dict[str, List[int]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for filename, value in (("G_c_leaves.json", good), ("B_c_leaves.json", bad)):
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
