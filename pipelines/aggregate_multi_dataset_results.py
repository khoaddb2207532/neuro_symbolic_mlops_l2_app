"""Validate and aggregate the complete 2-dataset x 5-seed experiment matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPECTED_METHODS = {
    "cnn_baseline",
    "gflownet_db",
    "random",
    "topk_confidence",
    "greedy_coverage",
    "gflownet_db_bayesian",
}


def _discover(input_root: Path) -> dict[str, tuple[Path, dict]]:
    runs: dict[str, tuple[Path, dict]] = {}
    for path in input_root.rglob("run_manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = str(manifest.get("run_id", "")).strip()
        if not run_id or manifest.get("status") != "complete" or not (path.parent / "results.csv").exists():
            continue
        previous = runs.get(run_id)
        if previous:
            keys = ("dataset_id", "dataset_fingerprint", "seed", "backbone", "git_commit")
            if any(previous[1].get(key) != manifest.get(key) for key in keys):
                raise RuntimeError(f"Conflicting copies found for run_id={run_id}")
            continue
        runs[run_id] = (path.parent, manifest)
    return runs


def aggregate(input_root: Path, registry_path: Path, output_dir: Path) -> None:
    registry = pd.read_csv(registry_path, dtype={"seed": int})
    if registry["run_id"].duplicated().any():
        raise RuntimeError("experiment_registry.csv contains duplicate run_id values")
    expected_ids = set(registry["run_id"])
    runs = _discover(input_root.resolve())
    found_ids = set(runs)
    if found_ids != expected_ids:
        raise RuntimeError(
            f"Experiment matrix is incomplete: missing={sorted(expected_ids - found_ids)}, "
            f"unexpected={sorted(found_ids - expected_ids)}"
        )

    audit_rows, result_frames = [], []
    for row in registry.itertuples(index=False):
        folder, manifest = runs[row.run_id]
        for key, expected in (
            ("dataset_id", row.dataset_id),
            ("seed", int(row.seed)),
            ("backbone", row.backbone),
            ("account_id", row.account_id),
        ):
            if str(manifest.get(key)) != str(expected):
                raise RuntimeError(f"{row.run_id}: manifest {key}={manifest.get(key)!r}, expected {expected!r}")
        frame = pd.read_csv(folder / "results.csv")
        if "method" not in frame.columns:
            raise RuntimeError(f"{row.run_id}: results.csv has no method column")
        methods = set(frame["method"].astype(str))
        if methods != EXPECTED_METHODS or len(frame) != len(EXPECTED_METHODS):
            raise RuntimeError(f"{row.run_id}: expected exactly six methods, got {sorted(methods)}")
        frame.insert(0, "run_id", row.run_id)
        frame.insert(1, "dataset_id", row.dataset_id)
        frame["seed"] = int(row.seed)
        result_frames.append(frame)
        audit_rows.append(
            {
                "run_id": row.run_id,
                "dataset_id": row.dataset_id,
                "seed": int(row.seed),
                "account_id": row.account_id,
                "dataset_fingerprint": manifest["dataset_fingerprint"],
                "git_commit": manifest.get("git_commit", "unknown"),
                "method_count": len(frame),
                "status": "validated",
            }
        )

    audit = pd.DataFrame(audit_rows)
    for dataset_id, group in audit.groupby("dataset_id"):
        if group["dataset_fingerprint"].nunique() != 1:
            raise RuntimeError(f"{dataset_id}: multiple dataset fingerprints found")
        if group["seed"].nunique() != 5:
            raise RuntimeError(f"{dataset_id}: expected five unique seeds")

    all_results = pd.concat(result_frames, ignore_index=True)
    if len(all_results) != 60:
        raise RuntimeError(f"Expected 60 result rows, got {len(all_results)}")
    numeric_metrics = [
        name for name in ("test_accuracy", "test_f1_macro")
        if name in all_results.columns
    ]
    if len(numeric_metrics) < 2:
        raise RuntimeError(
            "results.csv must contain full-precision test_accuracy and test_f1_macro"
        )

    summary = (
        all_results.groupby(["dataset_id", "method"])[numeric_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(part for part in col if part).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]

    paired_rows = []
    for dataset_id, dataset_frame in all_results.groupby("dataset_id"):
        pivot = dataset_frame.pivot(index="seed", columns="method", values=numeric_metrics)
        for metric in numeric_metrics:
            for method in sorted(EXPECTED_METHODS - {"cnn_baseline"}):
                delta = pivot[(metric, method)] - pivot[(metric, "cnn_baseline")]
                paired_rows.append({
                    "dataset_id": dataset_id,
                    "comparison": f"{method}_vs_baseline",
                    "metric": metric,
                    "mean_paired_delta": delta.mean(),
                    "sample_std_paired_delta": delta.std(ddof=1),
                    "n_seeds": len(delta),
                })
            delta = pivot[(metric, "gflownet_db_bayesian")] - pivot[(metric, "gflownet_db")]
            paired_rows.append({
                "dataset_id": dataset_id,
                "comparison": "bayesian_vs_gflownet_elite",
                "metric": metric,
                "mean_paired_delta": delta.mean(),
                "sample_std_paired_delta": delta.std(ddof=1),
                "n_seeds": len(delta),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "multi_dataset_run_audit.csv", index=False)
    all_results.to_csv(output_dir / "multi_dataset_all_results.csv", index=False)
    summary.to_csv(output_dir / "dataset_method_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(output_dir / "dataset_paired_deltas.csv", index=False)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_count": int(audit["dataset_id"].nunique()),
        "run_count": len(audit),
        "result_row_count": len(all_results),
        "methods": sorted(EXPECTED_METHODS),
        "note": "Statistics are computed per dataset; datasets are never pooled.",
    }
    (output_dir / "multi_dataset_aggregate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Validated {len(audit)} runs and {len(all_results)} method rows")
    print(f"Aggregate output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--registry", type=Path, default=Path("experiments/experiment_registry.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/multi_dataset_summary"))
    args = parser.parse_args()
    aggregate(args.input_root, args.registry, args.output_dir)


if __name__ == "__main__":
    main()
