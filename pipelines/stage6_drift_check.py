"""Stage 6/6: checkpoint drift gate, final comparison, and log-only ablation."""
import argparse
import csv
import glob
import json
import os
import re
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.data.dataset import create_dataloaders
from src.evaluation.evaluate import evaluate_classification_metrics
from src.evaluation.final_report import build_sensitivity_plots, sum_logged_duration
from src.models.cnn import (
    build_selected_baseline,
    selected_baseline_checkpoint,
    selected_baseline_metrics,
)
from src.rules.extractor import RuleExtractor
from src.rules.gfn_distribution_penalty import GFlowNetDistributionPenalty
from src.training.centroid import centroid_push_pull_loss, class_centroids
from src.training.drift import drift_decision, select_relative_drift_threshold, weighted_fidelity
from src.utils.checkpoint import load_model_weights
from src.utils.config import load_params
from src.utils.seed import set_seed


def _load_model(path, params, device, scope="last_block"):
    model = build_selected_baseline(params, pretrained=False).to(device)
    load_model_weights(model, path, device, required=True)
    return model


def _features_and_predictions(model, loader, device):
    feature_batches, predictions = [], []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            logits, features = model(images.to(device))
            feature_batches.append(features.cpu().numpy())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return np.concatenate(feature_batches), np.asarray(predictions)


def _checkpoint_epoch(path):
    match = re.search(r"epoch_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1


def _write_csv(path, rows, exclusive=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with open(path, "x" if exclusive else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _bad_count(model, loader, penalty, device):
    active = set(); model.eval()
    with torch.no_grad():
        for images, labels in loader:
            _, features = model(images.to(device))
            active.update(penalty.hard_bad_leaf_keys(features, labels.to(device)))
    return len(active)


def _train_centroid(params, train_loader, val_loader, centroids, device):
    cfg = params["centroid_baseline"]
    model = _load_model(selected_baseline_checkpoint(params), params, device)
    for parameter in model.parameters(): parameter.requires_grad = False
    modules = model.regularization_modules(cfg["scope"])
    for module in modules:
        for parameter in module.parameters(): parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"])
    started = time.perf_counter(); best, best_accuracy = None, -1.0
    centers = centroids.to(device)
    for _ in range(cfg["epochs"]):
        model.train(); model.freeze_bn()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits, features = model(images)
            loss = F.cross_entropy(logits, labels) + cfg["weight"] * centroid_push_pull_loss(
                features, labels, centers, cfg["margin"])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        accuracy = evaluate_classification_metrics(model, val_loader, device)["accuracy"]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/cnn_centroid.pt")
    return model, time.perf_counter() - started


def _log_cost(config_path):
    if not os.path.exists(config_path): return 0.0
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    return sum_logged_duration(config.get("search_log", ""))


def _all_logged_cost(log_dir, patterns):
    return sum(sum_logged_duration(path) for pattern in patterns
               for path in glob.glob(os.path.join(log_dir, pattern)))


def _historical_refit_count(log_dir):
    count = 0
    for path in glob.glob(os.path.join(log_dir, "drift_check_*.csv")):
        try:
            with open(path, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            count += any(str(row.get("refit_required", "")).lower() == "true" for row in rows)
        except (OSError, csv.Error):
            continue
    return int(count)


def _metric(metrics, split, name):
    values = metrics.get(split, {})
    return values.get(name, values.get("f1_macro" if name == "f1" else name, ""))


def _write_summary(rows, plots, refit_count, baseline_checkpoint):
    checks = {
        "GD1": [baseline_checkpoint, "data/features_train.npy"],
        "GD2": ["checkpoints/rf_model.pkl", "reports/leaf_stats.csv"],
        "GD3": ["reports/G_c_leaves.json", "reports/B_c_leaves.json"],
        "GD4": ["checkpoints/gfn_good.pt", "checkpoints/gfn_bad.pt", "reports/leaf_probs.json"],
        "GD5": ["checkpoints/cnn_regularized.pt", "reports/regularize_metrics.json"],
        "GD6-A": ["configs/drift_thresholds.yaml"],
        "GD6-B": ["reports/ablation/final_comparison.csv"],
    }
    complete = all(all(os.path.exists(path) for path in paths) for paths in checks.values()) and len(plots) >= 5
    lines = ["# Final pipeline evaluation", "", f"Pipeline complete: **{complete}**", "",
             "## Checklist", ""]
    for name, paths in checks.items():
        ok = all(os.path.exists(path) for path in paths)
        lines.append(f"- [{'x' if ok else ' '}] {name}")
    lines += ["", "## Final comparison", "", "| configuration | val accuracy | test accuracy | |B_c| | compute seconds |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['configuration']} | {row['val_accuracy']} | {row['test_accuracy']} | "
                     f"{row['bad_leaf_count']} | {row['total_compute_seconds']:.3f} |")
    lines += ["", f"Periodic refits triggered: **{refit_count}**.",
              "Sensitivity figures were generated only from existing timestamped logs; experiments were not rerun."]
    with open("reports/ablation/summary.md", "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def main(params_path):
    params = load_params(params_path); set_seed(params["seed"])
    baseline_checkpoint = selected_baseline_checkpoint(params)
    baseline_metrics_path = selected_baseline_metrics(params)
    required = ["checkpoints/cnn_regularized.pt", "reports/regularize_metrics.json",
                baseline_checkpoint, "checkpoints/rf_model.pkl",
                "reports/leaf_probs.json", baseline_metrics_path,
                "data/features_train.npy", "data/labels_train.npy"]
    missing = [path for path in required if not os.path.exists(path)]
    if missing: raise FileNotFoundError(f"stage-6 inputs are missing: {missing}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, train_loader, val_loader, test_loader = create_dataloaders(
        params["data_dir"], params["batch_size"], params["num_workers"], params["seed"])
    rf = joblib.load("checkpoints/rf_model.pkl")
    with open("configs/rf_hyperparams.yaml", encoding="utf-8") as stream:
        fitted_fidelity = float(yaml.safe_load(stream)["selected_fidelity"])
    checkpoints = sorted(glob.glob("checkpoints/regularized_history/epoch_*.pt"), key=_checkpoint_epoch)
    if not checkpoints: checkpoints = ["checkpoints/cnn_regularized.pt"]
    curve = []
    for path in checkpoints:
        features, cnn_predictions = _features_and_predictions(_load_model(path, params, device), val_loader, device)
        curve.append(weighted_fidelity(cnn_predictions, rf.predict(features)))
    cfg = params["drift"]
    threshold = select_relative_drift_threshold(curve, fitted_fidelity,
                                                cfg["minimum_relative_drop"], cfg["maximum_relative_drop"])
    rows = []
    for path, fidelity in zip(checkpoints, curve):
        decision = drift_decision(fidelity, fitted_fidelity, threshold)
        rows.append({"checkpoint": path, "epoch": _checkpoint_epoch(path), "fidelity": fidelity,
                     "fitted_fidelity": fitted_fidelity, "relative_drop": decision["relative_drop"],
                     "threshold": threshold, "refit_required": decision["refit_required"]})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    drift_log = os.path.join(params["logs_dir"], f"drift_check_{stamp}.csv")
    _write_csv(drift_log, rows, exclusive=True)
    os.makedirs("configs", exist_ok=True); os.makedirs("reports", exist_ok=True)
    with open("configs/drift_thresholds.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump({"relative_drop_threshold": threshold,
                        "minimum_relative_drop": cfg["minimum_relative_drop"],
                        "maximum_relative_drop": cfg["maximum_relative_drop"],
                        "check_every_epochs": cfg["check_every_epochs"],
                        "fitted_fidelity": fitted_fidelity, "selection": "median + 3*MAD, clipped",
                        "drift_log": drift_log}, stream, sort_keys=False)
    triggered = [row for row in rows if row["refit_required"]]
    if triggered:
        with open("reports/refit_required.json", "w", encoding="utf-8") as stream:
            json.dump({"trigger": triggered[0], "required_order": [2, 3, 4, 5],
                       "reuse_gflownet_checkpoint": False}, stream, indent=2)
        raise RuntimeError("drift threshold exceeded: refit stages 2->3->4->5 with new features; old GFlowNet is forbidden")

    with open("reports/leaf_probs.json", encoding="utf-8") as stream: leaf_probs = json.load(stream)
    with open(baseline_metrics_path, encoding="utf-8") as stream: baseline = json.load(stream)
    with open("reports/regularize_metrics.json", encoding="utf-8") as stream: regularized_report = json.load(stream)
    rules = RuleExtractor().extract(rf)
    penalty = GFlowNetDistributionPenalty(rules, leaf_probs, params["num_classes"], 0, 0,
                                           params["regularize"]["routing_temperature"]).to(device)
    regularized = _load_model("checkpoints/cnn_regularized.pt", params, device)
    regularized_val = evaluate_classification_metrics(regularized, val_loader, device)
    regularized_test = evaluate_classification_metrics(regularized, test_loader, device)
    centers = class_centroids(torch.from_numpy(np.load("data/features_train.npy")).float(),
                              torch.from_numpy(np.load("data/labels_train.npy")).long(), params["num_classes"])
    centroid, centroid_seconds = _train_centroid(params, train_loader, val_loader, centers, device)
    centroid_val = evaluate_classification_metrics(centroid, val_loader, device)
    centroid_test = evaluate_classification_metrics(centroid, test_loader, device)
    one_time_cost = sum(_log_cost(path) for path in ["configs/rf_hyperparams.yaml",
                                                     "configs/gfn_hyperparams.yaml",
                                                     "configs/regularize_hyperparams.yaml"])
    periodic_cost = _all_logged_cost(params["logs_dir"],
                                     ["search_rf_*.csv", "search_gfn_*.csv", "search_regularize_*.csv"])
    refit_count = _historical_refit_count(params["logs_dir"])
    rows = [
        {"configuration": "cnn_baseline", "val_accuracy": _metric(baseline, "validation", "accuracy"),
         "val_f1": _metric(baseline, "validation", "f1"), "test_accuracy": _metric(baseline, "test", "accuracy"),
         "test_f1": _metric(baseline, "test", "f1"), "bad_leaf_count": regularized_report["bad_leaf_count_before"],
         "total_compute_seconds": 0.0, "refit_count": 0},
        {"configuration": "centroid_push_pull", "val_accuracy": centroid_val["accuracy"], "val_f1": centroid_val["f1_macro"],
         "test_accuracy": centroid_test["accuracy"], "test_f1": centroid_test["f1_macro"],
         "bad_leaf_count": _bad_count(centroid, val_loader, penalty, device),
         "total_compute_seconds": centroid_seconds, "refit_count": 0},
        {"configuration": "gfn_regularize_once", "val_accuracy": regularized_val["accuracy"], "val_f1": regularized_val["f1_macro"],
         "test_accuracy": regularized_test["accuracy"], "test_f1": regularized_test["f1_macro"],
         "bad_leaf_count": regularized_report["bad_leaf_count_after"],
         "total_compute_seconds": one_time_cost, "refit_count": 0},
        {"configuration": "gfn_regularize_periodic", "val_accuracy": regularized_val["accuracy"], "val_f1": regularized_val["f1_macro"],
         "test_accuracy": regularized_test["accuracy"], "test_f1": regularized_test["f1_macro"],
         "bad_leaf_count": regularized_report["bad_leaf_count_after"],
         "total_compute_seconds": periodic_cost, "refit_count": refit_count},
    ]
    os.makedirs("reports/ablation", exist_ok=True)
    _write_csv("reports/ablation/final_comparison.csv", rows)
    plots = build_sensitivity_plots(params["logs_dir"], "reports/ablation/hyperparam_sensitivity")
    _write_summary(rows, plots, refit_count, baseline_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="params.yaml")
    main(parser.parse_args().config)
