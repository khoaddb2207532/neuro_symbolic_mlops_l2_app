"""Stage 3/6 — Wilson fidelity filtering and per-class G_c/B_c classification."""
import argparse
import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.rules.filter_search import (fidelity_valley, select_filter_elbow,
                                     threshold_ablation, write_filter_search_csv)
from src.rules.leaf_analysis import classify_leaves, save_leaf_groups
from src.utils.config import load_params
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main(params_path: str) -> None:
    params = load_params(params_path)
    stats_path = os.path.join("reports", "leaf_stats.csv")
    if not os.path.exists(stats_path):
        raise FileNotFoundError("reports/leaf_stats.csv is required; complete stage 2 first")
    stats = pd.read_csv(stats_path)
    required = {"leaf_id", "fidelity", "precision", "coverage", "wilson_fidelity_lower",
                "precision_hits", "target_class", "class_precision_hits",
                "class_prediction_count"}
    missing = required - set(stats.columns)
    if stats.empty or missing:
        raise ValueError(f"leaf_stats.csv is empty or missing columns: {sorted(missing)}")

    os.makedirs("reports", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(stats.fidelity, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
    axis.set(xlabel="Leaf fidelity", ylabel="Number of leaves", title="RF leaf fidelity distribution")
    figure.tight_layout(); figure.savefig(os.path.join("reports", "fidelity_histogram.png"), dpi=180)
    plt.close(figure)

    cfg = params["rule_filter"]
    valley = fidelity_valley(stats.fidelity, cfg.get("histogram_bins", 20))
    random_floor = 1.0 / params["num_classes"] * cfg.get("random_floor_multiplier", 1.5)
    rows = threshold_ablation(stats, cfg["fidelity_thresholds"], cfg["min_coverages"])
    selected = select_filter_elbow(rows, random_floor, valley)
    for row in rows:
        row.update({"histogram_valley": valley, "random_floor": random_floor,
                    "selected": row["fidelity_threshold"] == selected["fidelity_threshold"]
                    and row["min_coverage"] == selected["min_coverage"]})
    log_path = write_filter_search_csv(params["logs_dir"], rows)
    thresholds = {"tau_fidelity": selected["fidelity_threshold"],
                  "n_min": selected["min_coverage"], "histogram_valley": valley,
                  "random_floor": random_floor, "selection_method": "balanced_elbow",
                  "search_log": log_path}
    with open(os.path.join("configs", "filter_thresholds.yaml"), "w", encoding="utf-8") as stream:
        yaml.safe_dump(thresholds, stream, sort_keys=False)

    labels = np.load(os.path.join("data", "labels_val.npy"), allow_pickle=False)
    good, bad = classify_leaves(stats, labels, selected["fidelity_threshold"],
                                selected["min_coverage"], cfg["significance"])
    good = {str(i): good.get(str(i), []) for i in range(params["num_classes"])}
    bad = {str(i): bad.get(str(i), []) for i in range(params["num_classes"])}
    save_leaf_groups(good, bad, "reports")
    classes_with_good = sum(bool(good.get(str(i))) for i in range(params["num_classes"]))
    required_good = int(np.ceil(params["num_classes"] * cfg.get("min_good_class_fraction", 0.6)))
    if classes_with_good < required_good:
        raise RuntimeError(f"G_c is populated for only {classes_with_good}/{params['num_classes']} classes")
    if not any(bad.values()):
        raise RuntimeError("B_c is empty for every class; inspect thresholds/CNN errors before stage 4")

    # Legacy mirrors keep existing consumers working; canonical artifacts live in reports/.
    legacy = os.path.join(params["output_dir"], "03b_filtered_leaves")
    os.makedirs(legacy, exist_ok=True)
    shutil.copy2(stats_path, os.path.join(legacy, "leaf_stats.csv"))
    shutil.copy2(os.path.join("reports", "G_c_leaves.json"), os.path.join(legacy, "G_c_leaves.json"))
    shutil.copy2(os.path.join("reports", "B_c_leaves.json"), os.path.join(legacy, "B_c_leaves.json"))
    with open(os.path.join(legacy, "filter_search.json"), "w", encoding="utf-8") as stream:
        json.dump({"trials": rows, "selected": thresholds}, stream, indent=2)
    logger.info("Rule filter complete: tau=%.2f n_min=%d | G=%d B=%d",
                thresholds["tau_fidelity"], thresholds["n_min"],
                sum(map(len, good.values())), sum(map(len, bad.values())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    main(parser.parse_args().config)
