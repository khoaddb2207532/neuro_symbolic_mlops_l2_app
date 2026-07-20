"""Stage 5/6 — fine-tune CNN with frozen p_good/p_bad leaf distributions."""
import argparse
import copy
import csv
import itertools
import json
import os
import time
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml

from src.data.dataset import create_dataloaders
from src.evaluation.evaluate import evaluate_classification_metrics
from src.models.cnn import CNNBaseline
from src.rules.extractor import RuleExtractor
from src.rules.gfn_distribution_penalty import GFlowNetDistributionPenalty
from src.training.pareto import choose_conservative_pareto, pareto_front
from src.utils.checkpoint import load_model_weights
from src.utils.config import load_params
from src.utils.seed import set_seed


def _set_scope(model, scope):
    for parameter in model.parameters():
        parameter.requires_grad = False
    if scope == "last_layer":
        modules = [model.backbone.classifier[3]]
    elif scope == "last_two_layers":
        modules = [model.backbone.classifier[0], model.backbone.classifier[3]]
    elif scope == "full_backbone":
        modules = [model.backbone]
    else:
        raise ValueError(f"unknown regularization scope: {scope}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.freeze_bn()


def _active_bad_count(model, loader, penalty, device):
    model.eval(); active = set()
    with torch.no_grad():
        for images, labels in loader:
            _, features = model(images.to(device))
            active.update(penalty.hard_bad_leaf_keys(features, labels.to(device)))
    return len(active)


def _run_trial(params, checkpoint, rules, leaf_probs, train_loader, val_loader,
               device, alpha, beta, lr, scope, epochs, checkpoint_dir=None,
               checkpoint_interval=None):
    started = time.perf_counter()
    model = CNNBaseline(params["num_classes"], freeze_stage="last_block")
    load_model_weights(model, checkpoint, device, required=True)
    model = model.to(device); _set_scope(model, scope)
    penalty = GFlowNetDistributionPenalty(
        rules, leaf_probs, params["num_classes"], alpha, beta,
        temperature=params["regularize"]["routing_temperature"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
        weight_decay=params["weight_decay"],
    )
    best_state, best_bad, best_accuracy = copy.deepcopy(model.state_dict()), float("inf"), -1.0
    best_key = (float("inf"), float("inf"))
    bad_stale = accuracy_stale = 0
    history = []
    for epoch in range(epochs):
        model.train(); model.freeze_bn()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits, features = model(images)
            loss = F.cross_entropy(logits, labels) + penalty(features, logits, labels)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        accuracy = evaluate_classification_metrics(model, val_loader, device)["accuracy"]
        bad_count = _active_bad_count(model, val_loader, penalty, device)
        history.append({"epoch": epoch + 1, "accuracy": accuracy, "bad_leaf_count": bad_count})
        if checkpoint_dir and checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch + 1:04d}.pt"))
        current_key = (bad_count, -accuracy)
        if current_key < best_key:
            best_key = current_key
            best_state = copy.deepcopy(model.state_dict())
        if bad_count < best_bad:
            best_bad, bad_stale = bad_count, 0
        else:
            bad_stale += 1
        if accuracy > best_accuracy:
            best_accuracy, accuracy_stale = accuracy, 0
        else:
            accuracy_stale += 1
        if bad_stale >= params["regularize"]["bad_patience"] or \
                accuracy_stale >= params["regularize"]["accuracy_patience"]:
            break
    model.load_state_dict(best_state)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "selected.pt"))
    final_accuracy = evaluate_classification_metrics(model, val_loader, device)["accuracy"]
    final_bad = _active_bad_count(model, val_loader, penalty, device)
    return model, {"alpha": alpha, "beta": beta, "lr": lr, "scope": scope,
                   "epochs": len(history), "accuracy": final_accuracy,
                   "bad_leaf_count": final_bad,
                   "duration_seconds": time.perf_counter() - started}, history


def _write_log(records, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = os.path.join(log_dir, f"search_regularize_{stamp}.csv")
    fields = sorted({key for row in records for key in row})
    with open(path, "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    return path


def main(params_path):
    params = load_params(params_path); set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    required = ["reports/leaf_probs.json", "checkpoints/rf_model.pkl"]
    checkpoint = params["regularize"].get("input_checkpoint", "checkpoints/cnn_baseline.pt")
    required.append(checkpoint)
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"stage-5 inputs are missing: {missing}")
    with open("reports/leaf_probs.json", encoding="utf-8") as stream:
        leaf_probs = json.load(stream)
    rules = RuleExtractor().extract(joblib.load("checkpoints/rf_model.pkl"))
    _, train_loader, val_loader, _ = create_dataloaders(
        params["data_dir"], params["batch_size"], params["num_workers"], params["seed"])
    with open("reports/baseline_metrics.json", encoding="utf-8") as stream:
        baseline_accuracy = json.load(stream)["validation"]["accuracy"]
    cfg = params["regularize"]
    base_lr = cfg.get("base_lr", params["transfer_learning"]["lr_backbone"])

    baseline_model = CNNBaseline(params["num_classes"], freeze_stage="last_block").to(device)
    load_model_weights(baseline_model, checkpoint, device, required=True)
    baseline_penalty = GFlowNetDistributionPenalty(rules, leaf_probs, params["num_classes"],
                                                   0.0, 0.0, cfg["routing_temperature"]).to(device)
    bad_before = _active_bad_count(baseline_model, val_loader, baseline_penalty, device)
    records = []
    for alpha, beta in itertools.product(cfg["alpha_values"], cfg["beta_values"]):
        model, record, _ = _run_trial(
            params, checkpoint, rules, leaf_probs, train_loader, val_loader, device,
            alpha, beta, base_lr * cfg["alpha_beta_lr_ratio"], cfg["alpha_beta_scope"],
            cfg["search_epochs"])
        record["phase"] = "alpha_beta"; records.append(record)
    front = pareto_front(records, baseline_accuracy, cfg["max_accuracy_drop"])
    for record in records:
        record["pareto"] = record in front
    selected_ab = choose_conservative_pareto(front)

    scope_records = []
    for ratio, scope in itertools.product(cfg["lr_ratios"], cfg["scopes"]):
        _, record, _ = _run_trial(
            params, checkpoint, rules, leaf_probs, train_loader, val_loader, device,
            selected_ab["alpha"], selected_ab["beta"], base_lr * ratio, scope,
            cfg["search_epochs"])
        record.update({"phase": "lr_scope", "lr_ratio": ratio, "pareto": False})
        records.append(record); scope_records.append(record)
    feasible_scope = pareto_front(scope_records, baseline_accuracy, cfg["max_accuracy_drop"])
    selected_scope = choose_conservative_pareto(feasible_scope)
    final_model, final_record, history = _run_trial(
        params, checkpoint, rules, leaf_probs, train_loader, val_loader, device,
        selected_ab["alpha"], selected_ab["beta"], selected_scope["lr"],
        selected_scope["scope"], cfg["max_epochs"],
        checkpoint_dir="checkpoints/regularized_history",
        checkpoint_interval=params["drift"]["check_every_epochs"])
    final_record.update({"phase": "final", "pareto": False}); records.append(final_record)
    log_path = _write_log(records, params["logs_dir"])

    os.makedirs("reports", exist_ok=True); os.makedirs("configs", exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter([r["bad_leaf_count"] for r in records if r["phase"] == "alpha_beta"],
                 [r["accuracy"] for r in records if r["phase"] == "alpha_beta"], alpha=.5)
    axis.plot([r["bad_leaf_count"] for r in front], [r["accuracy"] for r in front], "r-o")
    axis.set(xlabel="|B_c| active leaves", ylabel="Validation accuracy", title="Regularization Pareto front")
    figure.tight_layout(); figure.savefig("reports/regularize_pareto.png", dpi=180); plt.close(figure)

    accuracy_after, bad_after = final_record["accuracy"], final_record["bad_leaf_count"]
    reduction_achieved = bad_after < bad_before
    if not reduction_achieved:
        logger_message = (
            "regularization did not reduce |B_c| (%d -> %d); stage continues in "
            "warning mode and records the unmet objective. Inspect %s"
        )
        # A single B leaf makes the required reduction a brittle 1 -> 0 test;
        # do not misreport success or discard the accuracy-preserving model.
        warnings.warn(logger_message % (bad_before, bad_after, log_path), RuntimeWarning)
    if baseline_accuracy - accuracy_after > cfg["max_accuracy_drop"]:
        raise RuntimeError("regularized CNN exceeds the predefined accuracy-drop budget")
    torch.save(final_model.state_dict(), "checkpoints/cnn_regularized.pt")
    selected = {"alpha": selected_ab["alpha"], "beta": selected_ab["beta"],
                "lr": selected_scope["lr"], "scope": selected_scope["scope"],
                "epochs": final_record["epochs"], "max_accuracy_drop": cfg["max_accuracy_drop"],
                "search_log": log_path}
    with open("configs/regularize_hyperparams.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(selected, stream, sort_keys=False)
    with open("reports/regularize_metrics.json", "w", encoding="utf-8") as stream:
        json.dump({"bad_leaf_count_before": bad_before, "bad_leaf_count_after": bad_after,
                   "bad_leaf_count_reduction": bad_before - bad_after,
                   "reduction_achieved": reduction_achieved,
                   "regularization_status": "reduced" if reduction_achieved else "no_reduction_warning",
                   "accuracy_before": baseline_accuracy, "accuracy_after": accuracy_after,
                   "accuracy_drop": baseline_accuracy - accuracy_after,
                   "epochs": final_record["epochs"]}, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="params.yaml")
    main(parser.parse_args().config)
