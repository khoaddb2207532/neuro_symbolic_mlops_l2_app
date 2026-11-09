"""Hậu kiểm công bằng checkpoint GFlowNet converged và diverse.

Mỗi checkpoint được sample cùng số ruleset, cùng repeat seeds và cùng batch
protocol. Không train hoặc cập nhật policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from src.gflownet.rule_ranking_analysis import (
    load_rule_order,
    rebuild_gflownet,
)
from src.utils.config import load_params
from src.utils.seed import set_seed


CHECKPOINTS = {
    "converged": "gflownet_best_converged.pth",
    "diverse": "gflownet_best_diverse.pth",
}
METRICS = (
    "mean_log_reward",
    "unique_ratio",
    "entropy",
    "entropy_normalized",
    "top1_share",
    "mean_ruleset_size",
)


def _write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def _sample_states(
    gflownet,
    env,
    n_samples: int,
    sample_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = []
    log_rewards = []
    sampled = 0
    while sampled < n_samples:
        batch_size = min(sample_batch_size, n_samples - sampled)
        trajectories = gflownet.sample_trajectories(
            env,
            n=batch_size,
            save_logprobs=True,
        )
        states.append(
            trajectories.terminating_states.tensor.bool().cpu()
        )
        log_rewards.append(trajectories.log_rewards.float().cpu())
        sampled += batch_size
    return torch.cat(states, dim=0), torch.cat(log_rewards, dim=0)


def _distribution_metrics(
    states: torch.Tensor,
    log_rewards: torch.Tensor,
) -> Dict[str, float]:
    n_samples = states.shape[0]
    _, counts = torch.unique(
        states,
        dim=0,
        return_counts=True,
    )
    probabilities = counts.float() / n_samples
    entropy = -(
        probabilities * probabilities.clamp(min=1e-12).log()
    ).sum()
    entropy_normalized = (
        float(entropy.item()) / math.log(n_samples)
        if n_samples > 1
        else 0.0
    )
    return {
        "mean_log_reward": float(log_rewards.mean().item()),
        "unique_ratio": float(counts.numel() / n_samples),
        "entropy": float(entropy.item()),
        "entropy_normalized": entropy_normalized,
        "top1_share": float(counts.max().item() / n_samples),
        "mean_ruleset_size": float(
            states.float().sum(dim=1).mean().item()
        ),
    }


def _sample_std(values: List[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def run(
    config_path: str,
    repeats: int,
    samples_per_repeat: int,
    sample_batch_size: int,
) -> Dict:
    if repeats < 2:
        raise ValueError("repeats phải >= 2 để tính sample standard deviation.")
    if samples_per_repeat <= 0 or sample_batch_size <= 0:
        raise ValueError("Số mẫu và batch size phải dương.")

    params = load_params(config_path)
    output_dir = Path(params["output_dir"]) / "04_filtered_rules"
    checkpoint_paths = {
        name: output_dir / filename
        for name, filename in CHECKPOINTS.items()
    }
    missing = [
        str(path)
        for path in checkpoint_paths.values()
        if not path.exists()
    ]
    status_path = output_dir / "checkpoint_posterior_evaluation_status.json"
    if missing:
        status = {
            "comparison_available": False,
            "reason": "missing_checkpoint",
            "missing_checkpoints": missing,
            "reporting_instruction": (
                "Không báo cáo bảng converged-vs-diverse. Chỉ nêu Bayesian "
                "dùng diverse checkpoint."
            ),
        }
        status_path.write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(status["reporting_instruction"])
        return status

    rule_order = load_rule_order(str(output_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repeat_rows = []
    base_seed = int(params["seed"]) * 10_000

    for checkpoint_name, checkpoint_path in checkpoint_paths.items():
        gflownet, env, _, _ = rebuild_gflownet(
            rule_order,
            str(checkpoint_path),
            device,
        )
        for repeat in range(repeats):
            # Hai checkpoint dùng cùng seed tại cùng repeat để phép so sánh
            # hậu nghiệm có protocol ngẫu nhiên đối xứng.
            repeat_seed = base_seed + repeat
            set_seed(repeat_seed)
            states, log_rewards = _sample_states(
                gflownet,
                env,
                samples_per_repeat,
                sample_batch_size,
            )
            metrics = _distribution_metrics(states, log_rewards)
            repeat_rows.append(
                {
                    "seed": int(params["seed"]),
                    "backbone": params["baseline_comparison"][
                        "selected_architecture"
                    ],
                    "loss_type": rule_order["loss_type"],
                    "checkpoint": checkpoint_name,
                    "repeat": repeat + 1,
                    "repeat_seed": repeat_seed,
                    "n_samples": samples_per_repeat,
                    **metrics,
                }
            )
        del gflownet, env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows = []
    for checkpoint_name in CHECKPOINTS:
        checkpoint_rows = [
            row
            for row in repeat_rows
            if row["checkpoint"] == checkpoint_name
        ]
        summary = {
            "seed": int(params["seed"]),
            "backbone": params["baseline_comparison"][
                "selected_architecture"
            ],
            "loss_type": rule_order["loss_type"],
            "checkpoint": checkpoint_name,
            "n_repeats": repeats,
            "samples_per_repeat": samples_per_repeat,
            "total_samples": repeats * samples_per_repeat,
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in checkpoint_rows]
            summary[metric + "_mean"] = float(np.mean(values))
            summary[metric + "_sample_std"] = _sample_std(values)
        summary_rows.append(summary)

    repeats_path = (
        output_dir / "checkpoint_posterior_evaluation_repeats.csv"
    )
    summary_csv_path = (
        output_dir / "checkpoint_posterior_evaluation_summary.csv"
    )
    summary_json_path = (
        output_dir / "checkpoint_posterior_evaluation_summary.json"
    )
    summary_text_path = (
        output_dir / "checkpoint_posterior_evaluation_summary.txt"
    )
    _write_csv(repeats_path, repeat_rows)
    _write_csv(summary_csv_path, summary_rows)
    summary_json_path.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with summary_text_path.open("w", encoding="utf-8") as file:
        file.write(
            f"Protocol: {repeats} repeats x "
            f"{samples_per_repeat} rulesets/checkpoint\n"
        )
        for row in summary_rows:
            file.write(f"\n[{row['checkpoint']}]\n")
            for metric in METRICS:
                file.write(
                    f"{metric} = {row[metric + '_mean']:.6f} ± "
                    f"{row[metric + '_sample_std']:.6f}\n"
                )

    status = {
        "comparison_available": True,
        "repeats": repeats,
        "samples_per_repeat": samples_per_repeat,
        "total_samples_per_checkpoint": repeats * samples_per_repeat,
        "summary_csv": str(summary_csv_path),
    }
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nCHECKPOINT POSTERIOR EVALUATION")
    for row in summary_rows:
        print(f"\n[{row['checkpoint']}]")
        for metric in METRICS:
            print(
                f"{metric}: {row[metric + '_mean']:.6f} ± "
                f"{row[metric + '_sample_std']:.6f}"
            )
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--samples-per-repeat", type=int, default=1000)
    parser.add_argument("--sample-batch-size", type=int, default=250)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(
        arguments.config,
        arguments.repeats,
        arguments.samples_per_repeat,
        arguments.sample_batch_size,
    )
