import numpy as np
import pandas as pd

from src.rules.filter_search import fidelity_valley, select_filter_elbow, threshold_ablation
from src.rules.leaf_analysis import classify_leaves


def test_histogram_finds_valley_between_two_fidelity_modes():
    values = np.concatenate([np.full(100, 0.25), np.full(100, 0.9)])
    valley = fidelity_valley(values, bins=20)
    assert valley is not None and 0.25 < valley < 0.9


def test_ablation_has_all_threshold_coverage_combinations():
    stats = pd.DataFrame({"wilson_fidelity_lower": [0.65, 0.82, 0.95],
                          "coverage": [100, 50, 20]})
    rows = threshold_ablation(stats, [0.6, 0.8], [10, 30])
    assert len(rows) == 4
    selected = select_filter_elbow(rows, random_floor=0.1, valley=0.7)
    assert selected["fidelity_threshold"] > 0.7
    assert 0 <= selected["coverage_ratio"] <= 1


def test_good_bad_use_class_precision_baseline_not_class_prevalence():
    stats = pd.DataFrame([
        {"leaf_id": 1, "target_class": 0, "coverage": 20,
         "wilson_fidelity_lower": 0.95, "precision_hits": 18, "precision": 0.9,
         "class_precision_hits": 50, "class_prediction_count": 100},
        {"leaf_id": 2, "target_class": 0, "coverage": 20,
         "wilson_fidelity_lower": 0.95, "precision_hits": 2, "precision": 0.1,
         "class_precision_hits": 50, "class_prediction_count": 100},
    ])
    good, bad = classify_leaves(stats, np.zeros(100), 0.8, 10, significance=0.05)
    assert good == {"0": [1]}
    assert bad == {"0": [2]}
