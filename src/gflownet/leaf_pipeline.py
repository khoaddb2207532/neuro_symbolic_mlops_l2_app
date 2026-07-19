"""Train exactly two torchgfn models: one over G leaves and one over B leaves."""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Dict, Mapping

import pandas as pd
import torch
import yaml

from src.gflownet.leaf_path import train_leaf_path_gflownet
from src.gflownet.validation import validate_leaf_probability_coverage
from src.rules.rule_types import RuleSet


def leaf_rewards(stats: pd.DataFrame, leaf_ids, coverage_power: float) -> Dict[int, float]:
    indexed = stats.set_index("leaf_id")
    maximum_coverage = max(float(stats.coverage.max()), 1.0)
    return {int(i): max(float(indexed.loc[i, "fidelity"])
                        * float(indexed.loc[i, "precision"])
                        * (float(indexed.loc[i, "coverage"]) / maximum_coverage) ** coverage_power,
                        1e-12) for i in leaf_ids}


def _timestamped_csv(log_dir: str, records: list) -> str:
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = os.path.join(log_dir, f"search_gfn_{stamp}.csv")
    fields = sorted({key for row in records for key in row})
    with open(path, "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    return path


def _normalize_by_class(probabilities, groups):
    result = {}
    for class_id, leaf_ids in sorted(groups.items(), key=lambda item: int(item[0])):
        ids = [int(i) for i in leaf_ids]
        denominator = sum(float(probabilities[i]) for i in ids)
        result[str(class_id)] = {str(i): float(probabilities[i]) / denominator for i in ids}
    return result


def train_two_leaf_gflownets(raw_rules: RuleSet, stats: pd.DataFrame,
                             good_groups: Mapping[str, list], bad_groups: Mapping[str, list],
                             config: Mapping, output_dir: str, logs_dir: str,
                             seed: int, device: str) -> Dict:
    """One good and one bad model; class-conditional tables are normalized views."""
    import optuna

    os.makedirs(output_dir, exist_ok=True); os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("configs", exist_ok=True); os.makedirs("reports", exist_ok=True)
    all_probabilities, chosen, records, summaries = {"good": {}, "bad": {}}, {}, [], []
    for kind, groups in (("good", good_groups), ("bad", bad_groups)):
        leaf_ids = sorted({int(i) for values in groups.values() for i in values})
        if not leaf_ids:
            raise RuntimeError(f"cannot train {kind} GFlowNet with an empty leaf universe")
        universe = {i: raw_rules.rules[i] for i in leaf_ids}
        rewards = leaf_rewards(stats, leaf_ids, config.get("coverage_power", 0.5))
        search = config["search"]
        best = {}

        def objective(trial):
            trial_cfg = {
                "hidden_dim": trial.suggest_categorical("hidden_dim", search["hidden_dims"]),
                "n_layers": trial.suggest_categorical("n_layers", search["n_layers"]),
                "lr": trial.suggest_categorical("lr", search["learning_rates"]),
                "exploration_rate": trial.suggest_categorical("exploration_rate", search["exploration_rates"]),
                "reward_beta": trial.suggest_categorical("reward_beta", search["reward_betas"]),
            }
            started = time.perf_counter()
            model, summary = train_leaf_path_gflownet(
                universe, rewards, hidden_dim=trial_cfg["hidden_dim"], n_layers=trial_cfg["n_layers"],
                learning_rate=trial_cfg["lr"], steps=config["num_iterations"],
                exploration=trial_cfg["exploration_rate"], reward_beta=trial_cfg["reward_beta"],
                kl_patience=config["kl_patience"], kl_tolerance=config["kl_tolerance"],
                seed=seed + trial.number + (10000 if kind == "bad" else 0), device=device,
                batch_size=config["batch_size"], logz_lr=config["logz_lr"])
            score = summary["best_kl"] + config.get("loss_stability_weight", 0.05) * summary["tb_loss_relative_std"]
            record = {"model": kind, "trial": trial.number, **trial_cfg, **summary,
                      "objective": score, "duration_seconds": time.perf_counter() - started}
            records.append(record)
            if score < best.get("score", float("inf")):
                best.update(score=score, model=model, summary=summary, config=trial_cfg)
            return score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=search["n_trials"])
        model, summary, selected = best["model"], best["summary"], best["config"]
        if not summary["converged"]:
            raise RuntimeError(f"{kind} GFlowNet did not reach KL plateau within the configured step cap")
        if summary["tb_loss_relative_std"] > config.get("max_tb_loss_relative_std", 0.5):
            raise RuntimeError(f"{kind} GFlowNet TB loss is unstable at the end of training")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        with torch.no_grad():
            raw_probs = {i: float(p.cpu()) for i, p in model.terminal_probabilities().items()}
        if set(raw_probs) != set(leaf_ids):
            raise RuntimeError(f"{kind} GFlowNet generated a terminal outside the filtered universe")
        all_probabilities[kind] = _normalize_by_class(raw_probs, groups)
        chosen[kind] = {**selected, **summary, "n_leaves": len(leaf_ids)}
        summaries.append({"model": kind, **summary})
        torch.save({"format": "torchgfn_leaf_path_v2", "kind": kind,
                    "state_dict": {name: value.cpu() for name, value in model.state_dict().items()},
                    "leaf_ids": leaf_ids, "rewards": rewards, "selected_config": selected},
                   os.path.join("checkpoints", f"gfn_{kind}.pt"))

    log_path = _timestamped_csv(logs_dir, records)
    validate_leaf_probability_coverage(all_probabilities, good_groups, bad_groups)
    chosen["search_log"] = log_path
    with open(os.path.join("configs", "gfn_hyperparams.yaml"), "w", encoding="utf-8") as stream:
        yaml.safe_dump(chosen, stream, sort_keys=False)
    with open(os.path.join("reports", "leaf_probs.json"), "w", encoding="utf-8") as stream:
        json.dump(all_probabilities, stream, indent=2)
    # Compatibility mirrors only; canonical outputs are checkpoints/configs/reports.
    for name, source in (("gfn_good.pt", "checkpoints/gfn_good.pt"),
                         ("gfn_bad.pt", "checkpoints/gfn_bad.pt")):
        shutil.copy2(source, os.path.join(output_dir, name))
    with open(os.path.join(output_dir, "leaf_probs.json"), "w", encoding="utf-8") as stream:
        json.dump(all_probabilities, stream, indent=2)
    with open(os.path.join(output_dir, "gfn_training_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2)
    return all_probabilities
