"""Stage 2/6 — tune RF by CNN fidelity, fit it, and measure every leaf."""
import argparse
import hashlib
import json
import os
import pickle

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier

from src.rules.extractor import RuleExtractor
from src.rules.leaf_analysis import build_forest_leaf_stats
from src.rules.rf_search import search_random_forest, write_timestamped_search_csv
from src.utils.config import load_params
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_array(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"required stage-1 artifact is missing: {path}")
    value = np.load(path, allow_pickle=False)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"invalid empty/NaN/Inf array: {path}")
    return value


def main(params_path: str) -> None:
    params = load_params(params_path)
    metadata_path = os.path.join("data", "features_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("feature provenance is missing; rerun stage 1 feature extraction")
    with open(metadata_path, encoding="utf-8") as stream:
        metadata = json.load(stream)
    source_checkpoint = metadata["cnn_checkpoint"]
    if not os.path.exists(source_checkpoint):
        raise FileNotFoundError(f"CNN checkpoint recorded by features no longer exists: {source_checkpoint}")
    digest = hashlib.sha256()
    with open(source_checkpoint, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != metadata["cnn_checkpoint_sha256"]:
        raise RuntimeError("CNN checkpoint changed after feature extraction; regenerate all features before RF refit")
    train_x = _load_array("data/features_train.npy")
    train_y = _load_array("data/labels_train.npy").reshape(-1)
    val_x = _load_array("data/features_val.npy")
    val_y = _load_array("data/labels_val.npy").reshape(-1)
    cnn_val = _load_array("data/cnn_predictions_val.npy").reshape(-1)
    if len(train_x) != len(train_y) or len(val_x) != len(val_y) or len(val_x) != len(cnn_val):
        raise ValueError("feature/label/CNN-prediction counts do not match")
    if len(train_x) != metadata["train_samples"] or len(val_x) != metadata["val_samples"]:
        raise ValueError("feature arrays do not match the sample counts recorded in provenance metadata")

    rf_cfg = params["rf"]
    default_cfg = rf_cfg["default_config"]
    default_model = RandomForestClassifier(**default_cfg, oob_score=True, bootstrap=True,
                                           random_state=params["seed"], n_jobs=-1)
    default_model.fit(train_x, train_y)
    default_fidelity = float(np.mean(default_model.predict(val_x) == cnn_val))

    candidate_model, candidate_selected, records = search_random_forest(
        train_x, train_y, val_x, cnn_val, rf_cfg["search_space"], params["seed"],
        rf_cfg.get("elbow_tolerance", 0.002),
    )
    candidate_fidelity = float(np.mean(candidate_model.predict(val_x) == cnn_val))
    if candidate_fidelity < default_fidelity:
        model = default_model
        selected = {**default_cfg, "weighted_fidelity": default_fidelity,
                    "oob_score": float(default_model.oob_score_)}
        selected_fidelity = default_fidelity
        selection_source = "default_fallback"
        logger.warning(
            "RF search candidate fidelity %.6f is below default %.6f; "
            "using the default RF instead of the worse searched model",
            candidate_fidelity, default_fidelity,
        )
    else:
        model = candidate_model
        selected = candidate_selected
        selected_fidelity = candidate_fidelity
        selection_source = "search"
    improvement = selected_fidelity - default_fidelity
    records.insert(0, {"phase": "default_baseline", **default_cfg,
                       "weighted_fidelity": default_fidelity,
                       "oob_score": float(default_model.oob_score_), "duration_seconds": 0.0})
    log_path = write_timestamped_search_csv(params["logs_dir"], records)
    if improvement < rf_cfg.get("min_fidelity_improvement", 0.0):
        logger.warning(
            "RF search improvement %.6f is below target %.6f; stage continues "
            "with the best non-degrading model. Search details: %s",
            improvement, rf_cfg["min_fidelity_improvement"], log_path,
        )

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    legacy_dir = os.path.join(params["output_dir"], "03_rules")
    os.makedirs(legacy_dir, exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", "rf_model.pkl")
    joblib.dump(model, checkpoint_path)
    joblib.dump(model, os.path.join(legacy_dir, "rf_model.joblib"))
    raw_rules = RuleExtractor().extract(model)
    with open(os.path.join(legacy_dir, "raw_rules.pkl"), "wb") as stream:
        pickle.dump(raw_rules, stream)
    stats = build_forest_leaf_stats(model, raw_rules, val_x, val_y, cnn_val)
    if stats.empty or not {"fidelity", "precision", "coverage"}.issubset(stats.columns):
        raise RuntimeError("leaf_stats.csv would be empty or incomplete")
    stats.to_csv(os.path.join("reports", "leaf_stats.csv"), index=False)
    chosen = {**selected, "default_fidelity": default_fidelity,
              "selected_fidelity": selected_fidelity, "fidelity_improvement": improvement,
              "searched_candidate_fidelity": candidate_fidelity,
              "selection_source": selection_source,
              "search_log": log_path}
    with open(os.path.join("configs", "rf_hyperparams.yaml"), "w", encoding="utf-8") as stream:
        yaml.safe_dump(chosen, stream, sort_keys=False, allow_unicode=True)
    with open(os.path.join(legacy_dir, "rf_best_config.json"), "w", encoding="utf-8") as stream:
        json.dump(chosen, stream, indent=2, default=str)
    logger.info("RF stage complete: fidelity %.6f -> %.6f; %d leaves",
                default_fidelity, selected_fidelity, len(stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    main(parser.parse_args().config)
