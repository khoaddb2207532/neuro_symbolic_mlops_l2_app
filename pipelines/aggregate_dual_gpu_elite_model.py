"""Aggregate one backbone's dual-GPU comparison CSVs across all seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable


METRICS = (
    "test_accuracy",
    "test_f1_macro",
    "accuracy_delta_vs_baseline",
    "f1_delta_vs_baseline",
)


def _write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Không có kết quả aggregate để ghi.")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    input_root: Path,
    output_dir: Path,
    backbone: str,
    expected_seeds: list[int],
) -> tuple[Path, Path, Path]:
    files = sorted(input_root.rglob("seed_*_dual_elite_comparison.csv"))
    records = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("backbone") != backbone:
                    continue
                row["seed"] = int(row["seed"])
                row["dataset_id"] = row.get("dataset_id") or "unknown"
                for metric in METRICS:
                    row[metric] = float(row[metric])
                records.append(row)
    if not records:
        raise FileNotFoundError(
            f"Không tìm thấy comparison CSV cho backbone={backbone} trong {input_root}."
        )

    grouped = defaultdict(list)
    for row in records:
        grouped[(row["dataset_id"], row["method"])].append(row)

    expected = set(expected_seeds)
    summary = []
    for (dataset_id, method), rows in sorted(grouped.items()):
        observed = {row["seed"] for row in rows}
        if observed != expected:
            raise ValueError(
                f"Thiếu seed cho {dataset_id}/{backbone}/{method}: "
                f"observed={sorted(observed)}, expected={sorted(expected)}"
            )
        item = {
            "dataset_id": dataset_id,
            "backbone": backbone,
            "method": method,
            "n_seeds": len(rows),
            "seeds": ",".join(map(str, sorted(observed))),
        }
        for metric in METRICS:
            values = [row[metric] for row in rows]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary.append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{backbone}_all_seeds_summary.csv"
    json_path = output_dir / f"{backbone}_all_seeds_summary.json"
    report_path = output_dir / f"{backbone}_all_seeds_summary.md"
    _write_csv(csv_path, summary)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ranked = sorted(
        summary,
        key=lambda row: (
            row["dataset_id"],
            -row["test_f1_macro_mean"],
        ),
    )
    lines = [
        f"# {backbone}: aggregate across seeds",
        "",
        "| Dataset | Method | Accuracy mean±std | Macro-F1 mean±std | ΔAcc vs baseline | ΔF1 vs baseline |",
        "|---|---|---:|---:|---:|---:|",
    ]
    print(f"\n{backbone} - AGGREGATE ACROSS SEEDS", flush=True)
    print(
        f"{'Dataset':<12} {'Method':<34} {'Accuracy':>17} {'Macro-F1':>17}",
        flush=True,
    )
    for row in ranked:
        accuracy = (
            f"{row['test_accuracy_mean']:.4f}±"
            f"{row['test_accuracy_std']:.4f}"
        )
        macro_f1 = (
            f"{row['test_f1_macro_mean']:.4f}±"
            f"{row['test_f1_macro_std']:.4f}"
        )
        print(
            f"{row['dataset_id']:<12} {row['method']:<34} "
            f"{accuracy:>17} {macro_f1:>17}",
            flush=True,
        )
        lines.append(
            f"| {row['dataset_id']} | {row['method']} | {accuracy} | "
            f"{macro_f1} | {row['accuracy_delta_vs_baseline_mean']:+.4f} | "
            f"{row['f1_delta_vs_baseline_mean']:+.4f} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    paths = aggregate(
        args.input_root,
        args.output_dir,
        args.backbone,
        args.expected_seeds,
    )
    print("\nAggregate outputs:")
    for path in paths:
        print(" -", path)
