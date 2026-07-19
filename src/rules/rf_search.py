"""Sequential RF tuning driven by CNN fidelity, with OOB diagnostics."""
from __future__ import annotations

import csv
import os
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _fit_trial(train_x, train_y, val_x, cnn_val_predictions, config, seed, phase):
    started = time.perf_counter()
    model = RandomForestClassifier(**config, oob_score=True, bootstrap=True,
                                   random_state=seed, n_jobs=-1)
    model.fit(train_x, train_y)
    fidelity = float(np.mean(model.predict(val_x) == cnn_val_predictions))
    return model, {"phase": phase, **config, "weighted_fidelity": fidelity,
                   "oob_score": float(model.oob_score_),
                   "duration_seconds": time.perf_counter() - started}


def _best_by_fidelity(trials):
    return max(trials, key=lambda item: (item[1]["weighted_fidelity"], item[1]["oob_score"]))


def search_random_forest(train_features: np.ndarray, train_labels: np.ndarray,
                         val_features: np.ndarray, val_cnn_predictions: np.ndarray,
                         search_space: Mapping, seed: int = 42,
                         elbow_tolerance: float = 0.002) -> Tuple[RandomForestClassifier, Dict, List[Dict]]:
    """Tune one RF dimension at a time so later phases use a frozen earlier choice."""
    selected = {
        "max_depth": search_space.get("initial_max_depth", None),
        "n_estimators": search_space.get("initial_n_estimators", 100),
        "min_samples_leaf": search_space.get("initial_min_samples_leaf", 1),
        "max_features": search_space.get("initial_max_features", "sqrt"),
    }
    records, final_model = [], None

    depth_trials = []
    for value in search_space["max_depth"]:
        config = {**selected, "max_depth": value}
        result = _fit_trial(train_features, train_labels, val_features, val_cnn_predictions,
                            config, seed, "max_depth")
        depth_trials.append(result); records.append(result[1])
    final_model, best = _best_by_fidelity(depth_trials)
    selected["max_depth"] = best["max_depth"]

    estimator_trials = []
    for value in sorted(search_space["n_estimators"]):
        config = {**selected, "n_estimators": value}
        result = _fit_trial(train_features, train_labels, val_features, val_cnn_predictions,
                            config, seed, "n_estimators")
        estimator_trials.append(result); records.append(result[1])
    max_fidelity = max(item[1]["weighted_fidelity"] for item in estimator_trials)
    elbow_candidates = [item for item in estimator_trials
                        if max_fidelity - item[1]["weighted_fidelity"] <= elbow_tolerance]
    final_model, best = min(elbow_candidates, key=lambda item: item[1]["n_estimators"])
    selected["n_estimators"] = best["n_estimators"]

    leaf_values = list(search_space["min_samples_leaf"])
    random.Random(seed).shuffle(leaf_values)
    leaf_values = leaf_values[: search_space.get("min_samples_leaf_trials", len(leaf_values))]
    leaf_trials = []
    for value in leaf_values:
        config = {**selected, "min_samples_leaf": value}
        result = _fit_trial(train_features, train_labels, val_features, val_cnn_predictions,
                            config, seed, "min_samples_leaf_random")
        leaf_trials.append(result); records.append(result[1])
    final_model, best = _best_by_fidelity(leaf_trials)
    selected["min_samples_leaf"] = best["min_samples_leaf"]

    feature_trials = []
    for value in search_space["max_features"]:
        config = {**selected, "max_features": value}
        result = _fit_trial(train_features, train_labels, val_features, val_cnn_predictions,
                            config, seed, "max_features_oob")
        feature_trials.append(result); records.append(result[1])
    top_fidelity = max(item[1]["weighted_fidelity"] for item in feature_trials)
    comparable = [item for item in feature_trials
                  if top_fidelity - item[1]["weighted_fidelity"] <= elbow_tolerance]
    final_model, best = max(comparable, key=lambda item: item[1]["oob_score"])
    selected["max_features"] = best["max_features"]

    return final_model, {**selected, "weighted_fidelity": best["weighted_fidelity"],
                         "oob_score": best["oob_score"]}, records


def write_timestamped_search_csv(log_dir: str, records: List[Dict]) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = os.path.join(log_dir, f"search_rf_{timestamp}.csv")
    fields = list(records[0])
    with open(path, "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    return path
