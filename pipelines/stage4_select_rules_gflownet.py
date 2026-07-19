"""Stage 4 — train leaf-path TB-GFlowNets over filtered RF terminals.

Exactly two models are trained independently (good and bad). The CNN is never
forwarded here: rewards come only from leaf_stats.csv.
"""
import argparse
import json
import os
import pickle

import joblib
import pandas as pd
import torch

from src.gflownet.leaf_pipeline import train_two_leaf_gflownets
from src.rules.extractor import RuleExtractor
from src.rules.io import save_rules_excel
from src.rules.rule_types import RuleSet
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _load_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def main(params_path: str) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = os.path.join(params["output_dir"], "04_filtered_rules")
    os.makedirs(output_dir, exist_ok=True)

    rf_path = os.path.join("checkpoints", "rf_model.pkl")
    if not os.path.exists(rf_path):
        raise FileNotFoundError("checkpoints/rf_model.pkl is required; complete stage 2 first")
    raw_rules = RuleExtractor().extract(joblib.load(rf_path))
    stats = pd.read_csv(os.path.join("reports", "leaf_stats.csv"))
    good = _load_json(os.path.join("reports", "G_c_leaves.json"))
    bad = _load_json(os.path.join("reports", "B_c_leaves.json"))
    admitted = sorted({int(i) for groups in (good, bad) for ids in groups.values() for i in ids})
    if not admitted:
        raise RuntimeError("No leaf passed stage 3b; tune rule_filter before GFlowNet training")
    good_ids = sorted({int(i) for ids in good.values() for i in ids})
    bad_ids = sorted({int(i) for ids in bad.values() for i in ids})
    if set(good_ids) & set(bad_ids):
        raise RuntimeError("a leaf cannot belong to both G_c and B_c")
    if any(i < 0 or i >= len(raw_rules.rules) for i in admitted):
        raise RuntimeError("filtered leaf id does not exist in checkpoints/rf_model.pkl")
    if not set(admitted).issubset(set(stats.leaf_id.astype(int))):
        raise RuntimeError("filtered leaf id is missing from reports/leaf_stats.csv")

    probabilities = train_two_leaf_gflownets(
        raw_rules, stats, good, bad, params["gflownet"], output_dir,
        params["logs_dir"], params["seed"], device,
    )

    # Explicit good/bad universes are the canonical downstream artifacts.
    good_rules = raw_rules.filter_rules(good_ids)
    bad_rules = raw_rules.filter_rules(bad_ids)
    valid_rules = raw_rules.filter_rules(admitted)
    for name, value in (("good_rules.pkl", good_rules), ("bad_rules.pkl", bad_rules),
                        ("valid_rules.pkl", valid_rules),
                        ("selected_rules.pkl", good_rules)):
        with open(os.path.join(output_dir, name), "wb") as stream:
            pickle.dump(value, stream)
    save_rules_excel(good_rules.rules, os.path.join(output_dir, "good_rules.xlsx"))
    save_rules_excel(bad_rules.rules, os.path.join(output_dir, "bad_rules.xlsx"))
    logger.info("Leaf-path GFlowNet complete: %d good leaves, %d bad leaves", len(good_ids), len(bad_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    main(parser.parse_args().config)
