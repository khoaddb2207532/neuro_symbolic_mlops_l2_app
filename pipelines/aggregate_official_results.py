"""Tạo bảng so sánh chính thức từ kết quả nhiều seed.

Output chính có một dòng cho mỗi (seed, method), đồng thời mang các thống kê
group-level để bảng duy nhất đủ dùng cho audit và báo cáo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


METHODS = (
    "cnn_baseline",
    "gflownet_db",
    "gflownet_db_bayesian",
    "random",
    "topk_confidence",
    "greedy_coverage",
)
METRICS = (
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_f1_macro",
    "test_weighted_f1",
)


def _result_files(input_root: Path) -> List[Path]:
    return sorted(
        path
        for path in input_root.rglob("seed_*_results.csv")
        if "_with_bayesian" not in path.name
    )


def _load_results(input_root: Path) -> pd.DataFrame:
    files = _result_files(input_root)
    if not files:
        raise FileNotFoundError(
            f"Không tìm thấy seed_*_results.csv trong {input_root}."
        )
    frames = []
    for order, path in enumerate(files):
        frame = pd.read_csv(path)
        frame["_source_order"] = order
        frame["_source_path"] = str(path)
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True, sort=False)
    required = {"seed", "backbone", "method", "test_accuracy", "test_f1_macro"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Result CSV thiếu cột: {sorted(missing)}")
    # File seed result mới chứa exact metrics và sáu phương pháp. Nếu Add Input
    # bị trùng, giữ candidate cuối theo thứ tự path ổn định.
    return (
        results.sort_values("_source_order")
        .drop_duplicates(["seed", "backbone", "method"], keep="last")
        .drop(columns=["_source_order", "_source_path"])
    )


def build_official_table(
    results: pd.DataFrame,
    expected_seeds: Iterable[int],
) -> pd.DataFrame:
    expected_seeds = tuple(sorted(set(int(seed) for seed in expected_seeds)))
    results = results[
        results["seed"].isin(expected_seeds)
        & results["method"].isin(METHODS)
    ].copy()

    backbones = set(results["backbone"].dropna().astype(str))
    if len(backbones) != 1:
        raise ValueError(
            f"Kết quả phải dùng đúng một backbone, nhận {sorted(backbones)}."
        )
    for seed in expected_seeds:
        present = set(results.loc[results["seed"] == seed, "method"])
        missing = set(METHODS) - present
        if missing:
            raise ValueError(
                f"Seed {seed} thiếu phương pháp: {sorted(missing)}"
            )
    if len(results) != len(expected_seeds) * len(METHODS):
        raise ValueError("Số dòng kết quả không đúng seed × method.")

    available_metrics = [metric for metric in METRICS if metric in results]
    for metric in available_metrics:
        results[metric] = pd.to_numeric(results[metric], errors="raise")

    # Mean và SAMPLE standard deviation (pandas std mặc định ddof=1).
    aggregations = {}
    for metric in available_metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_sample_std"] = (metric, "std")
    stats = (
        results.groupby(["backbone", "method"])
        .agg(n_seeds=("seed", "nunique"), **aggregations)
        .reset_index()
    )
    if not stats["n_seeds"].eq(len(expected_seeds)).all():
        raise AssertionError("Có phương pháp không đủ số seed yêu cầu.")
    official = results.merge(stats, on=["backbone", "method"], how="left")

    # Paired delta với baseline trong cùng seed.
    baseline = results.loc[
        results["method"] == "cnn_baseline",
        ["seed", *available_metrics],
    ].set_index("seed")
    for metric in available_metrics:
        delta_column = f"paired_{metric}_delta_vs_baseline"
        official[delta_column] = official.apply(
            lambda row: row[metric] - baseline.loc[row["seed"], metric],
            axis=1,
        )
        delta_stats = (
            official.groupby("method")[delta_column]
            .agg(["mean", "std"])
            .rename(
                columns={
                    "mean": delta_column + "_mean",
                    "std": delta_column + "_sample_std",
                }
            )
        )
        official = official.merge(
            delta_stats,
            left_on="method",
            right_index=True,
            how="left",
        )

    # Paired Bayesian - GFlowNet elite/fixed trong cùng seed.
    bayesian_mask = official["method"] == "gflownet_db_bayesian"
    elite = results.loc[
        results["method"] == "gflownet_db",
        ["seed", *available_metrics],
    ].set_index("seed")
    for metric in available_metrics:
        column = f"paired_{metric}_delta_bayesian_vs_gflownet_elite"
        official[column] = pd.NA
        official.loc[bayesian_mask, column] = official.loc[
            bayesian_mask
        ].apply(
            lambda row: row[metric] - elite.loc[row["seed"], metric],
            axis=1,
        )
        bayesian_values = pd.to_numeric(
            official.loc[bayesian_mask, column]
        )
        official.loc[bayesian_mask, column + "_mean"] = (
            bayesian_values.mean()
        )
        official.loc[bayesian_mask, column + "_sample_std"] = (
            bayesian_values.std(ddof=1)
        )

    return official.sort_values(["seed", "method"]).reset_index(drop=True)


def run(
    input_root: str,
    output_dir: str,
    expected_seeds: Iterable[int],
) -> Path:
    results = _load_results(Path(input_root))
    official = build_official_table(results, expected_seeds)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "official_experiment_comparison.csv"
    official.to_csv(output_path, index=False)
    print("Đã tạo bảng chính thức:", output_path)
    print(
        official[
            [
                "seed",
                "method",
                "test_accuracy",
                "test_f1_macro",
                "test_accuracy_mean",
                "test_accuracy_sample_std",
                "paired_test_accuracy_delta_vs_baseline",
            ]
        ].to_string(index=False)
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/combined_results",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=[42, 44, 46, 48, 50],
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(
        arguments.input_root,
        arguments.output_dir,
        arguments.expected_seeds,
    )
