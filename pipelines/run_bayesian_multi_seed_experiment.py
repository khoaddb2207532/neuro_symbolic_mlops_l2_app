"""Chạy/resume Bayesian Stage 5 cho nhiều seed và chỉ tổng hợp khi đủ."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from pipelines.run_core_seed_experiment import SUPPORTED_BACKBONES


CORE_METHODS = (
    "cnn_baseline",
    "gflownet_db",
    "random",
    "topk_confidence",
    "greedy_coverage",
)


def _run_seed(args: argparse.Namespace, seed: int) -> None:
    output_dir = Path(args.output_root) / f"outputs_seed_{seed}"
    command = [
        sys.executable,
        "-m",
        "pipelines.run_bayesian_seed_experiment",
        "--config",
        args.config,
        "--seed",
        str(seed),
        "--backbone",
        args.backbone,
        "--mc-samples",
        str(args.mc_samples),
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(output_dir),
        "--project-dir",
        args.project_dir,
        "--working-dir",
        args.working_dir,
        "--kaggle-input-root",
        args.kaggle_input_root,
        "--sampler-checkpoint",
        args.sampler_checkpoint,
        "--defer-comparison",
    ]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=args.project_dir, check=True)


def _read_csv(path: Path) -> List[Dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _core_rows(input_root: Path, seed: int, backbone: str) -> Dict[str, Dict]:
    candidates = [
        path
        for path in input_root.rglob(f"seed_{seed}_results.csv")
        if "_bayesian_" not in path.name and "_with_bayesian" not in path.name
    ]
    rows: Dict[str, Dict] = {}
    for path in sorted(candidates):
        for row in _read_csv(path):
            if int(row["seed"]) != seed:
                continue
            if row.get("backbone") and row["backbone"] != backbone:
                continue
            rows[row["method"]] = row
    missing = set(CORE_METHODS) - set(rows)
    if missing:
        raise ValueError(f"Seed {seed} thiếu kết quả lõi: {sorted(missing)}")
    return rows


def _mean_std(values: List[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def _write_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(args: argparse.Namespace) -> None:
    seeds = args.seeds
    working = Path(args.working_dir)
    input_root = Path(args.kaggle_input_root)
    per_seed_rows = []

    for seed in seeds:
        bayes_path = working / f"seed_{seed}_bayesian_results.csv"
        if not bayes_path.exists():
            raise FileNotFoundError(
                f"Chưa đủ dữ liệu: thiếu {bayes_path}. Không tổng hợp."
            )
        bayes = _read_csv(bayes_path)[0]
        if bayes["backbone"] != args.backbone:
            raise ValueError(f"Seed {seed} dùng backbone khác.")
        core = _core_rows(input_root, seed, args.backbone)
        for method in CORE_METHODS:
            reference = core[method]
            per_seed_rows.append(
                {
                    "seed": seed,
                    "backbone": args.backbone,
                    "comparison": f"gflownet_db_bayesian - {method}",
                    "bayesian_accuracy": float(bayes["test_accuracy"]),
                    "reference_accuracy": float(reference["test_accuracy"]),
                    "accuracy_delta": float(bayes["test_accuracy"])
                    - float(reference["test_accuracy"]),
                    "bayesian_f1_macro": float(bayes["test_f1_macro"]),
                    "reference_f1_macro": float(reference["test_f1_macro"]),
                    "f1_macro_delta": float(bayes["test_f1_macro"])
                    - float(reference["test_f1_macro"]),
                }
            )

    expected_count = len(seeds) * len(CORE_METHODS)
    if len(per_seed_rows) != expected_count:
        raise RuntimeError("Chưa đủ dữ liệu paired; không tổng hợp.")

    summary_rows = []
    for method in CORE_METHODS:
        label = f"gflownet_db_bayesian - {method}"
        rows = [row for row in per_seed_rows if row["comparison"] == label]
        accuracy = [row["accuracy_delta"] for row in rows]
        f1 = [row["f1_macro_delta"] for row in rows]
        acc_mean, acc_std = _mean_std(accuracy)
        f1_mean, f1_std = _mean_std(f1)
        summary_rows.append(
            {
                "backbone": args.backbone,
                "comparison": label,
                "n_seeds": len(rows),
                "accuracy_delta_mean": acc_mean,
                "accuracy_delta_std": acc_std,
                "f1_macro_delta_mean": f1_mean,
                "f1_macro_delta_std": f1_std,
            }
        )

    detail_path = working / "bayesian_comparison_all_seeds.csv"
    summary_path = working / "bayesian_comparison_mean_std.csv"
    _write_csv(detail_path, per_seed_rows)
    _write_csv(summary_path, summary_rows)

    print("\nĐÃ ĐỦ DỮ LIỆU TẤT CẢ SEED — KẾT QUẢ TỔNG HỢP")
    print(
        f"{'Comparison':<48} {'Δ Acc mean±std':>22} "
        f"{'Δ F1 mean±std':>22}"
    )
    for row in summary_rows:
        print(
            f"{row['comparison']:<48} "
            f"{row['accuracy_delta_mean']:>+9.4f} ± "
            f"{row['accuracy_delta_std']:<8.4f} "
            f"{row['f1_macro_delta_mean']:>+9.4f} ± "
            f"{row['f1_macro_delta_std']:<8.4f}"
        )
    print(" -", detail_path)
    print(" -", summary_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 44, 46, 48, 50],
    )
    parser.add_argument(
        "--backbone",
        default="mobilenetv3_small",
        choices=SUPPORTED_BACKBONES,
    )
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-dir", default=os.getcwd())
    parser.add_argument("--working-dir", default="/kaggle/working")
    parser.add_argument("--kaggle-input-root", default="/kaggle/input")
    parser.add_argument(
        "--sampler-checkpoint",
        default="gflownet_best_diverse.pth",
    )
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    for experiment_seed in parsed.seeds:
        _run_seed(parsed, experiment_seed)
    _aggregate(parsed)
