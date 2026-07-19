import csv

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.rules.extractor import RuleExtractor
from src.rules.leaf_analysis import build_forest_leaf_stats
from src.rules.rf_search import search_random_forest, write_timestamped_search_csv


def _dataset():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(80, 4))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    return x[:60], y[:60], x[60:], y[60:]


def test_rf_search_logs_all_sequential_phases(tmp_path):
    train_x, train_y, val_x, cnn_val = _dataset()
    space = {
        "initial_max_depth": None, "initial_n_estimators": 5,
        "initial_min_samples_leaf": 2, "initial_max_features": "sqrt",
        "max_depth": [2, None], "n_estimators": [5, 10],
        "min_samples_leaf": [1, 2], "min_samples_leaf_trials": 1,
        "max_features": ["sqrt", 2],
    }
    model, selected, records = search_random_forest(
        train_x, train_y, val_x, cnn_val, space, seed=2, elbow_tolerance=0.01
    )
    assert {record["phase"] for record in records} == {
        "max_depth", "n_estimators", "min_samples_leaf_random", "max_features_oob"
    }
    assert "weighted_fidelity" in selected and hasattr(model, "estimators_")
    first = write_timestamped_search_csv(str(tmp_path), records)
    second = write_timestamped_search_csv(str(tmp_path), records)
    assert first != second
    with open(first, encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == len(records)


def test_leaf_stats_cover_every_extracted_leaf():
    train_x, train_y, val_x, cnn_val = _dataset()
    model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=2).fit(train_x, train_y)
    rules = RuleExtractor().extract(model)
    stats = build_forest_leaf_stats(model, rules, val_x, cnn_val, cnn_val)
    assert len(stats) == len(rules)
    assert {"leaf_id", "fidelity", "precision", "coverage"}.issubset(stats.columns)
    assert stats.coverage.ge(0).all()
