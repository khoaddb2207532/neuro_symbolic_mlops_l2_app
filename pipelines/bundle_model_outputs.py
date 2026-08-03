"""Collect completed managed exports into a bundle for one model."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipelines.aggregate_official_results import build_official_table


IDENTITY_COLUMNS = ("run_id", "dataset_id", "seed", "backbone")
EXPECTED_METHODS = {
    "cnn_baseline",
    "gflownet_db",
    "gflownet_db_bayesian",
    "random",
    "topk_confidence",
    "greedy_coverage",
}
RULE_QUALITY_METHODS = {
    "gflownet_elite",
    "random",
    "topk_confidence",
    "greedy_coverage",
}


def _attach_identity(frame: pd.DataFrame, run_id: str, manifest: dict) -> pd.DataFrame:
    identity = {
        "run_id": run_id,
        "dataset_id": str(manifest["dataset_id"]),
        "seed": int(manifest["seed"]),
        "backbone": str(manifest["backbone"]),
    }
    for column, expected_value in identity.items():
        if column in frame.columns:
            observed = frame[column].dropna()
            if column == "seed":
                try:
                    observed_values = set(pd.to_numeric(observed).astype(int))
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"{run_id}: results.csv contains invalid seed values"
                    ) from exc
                expected_values = {int(expected_value)}
            else:
                observed_values = set(observed.astype(str))
                expected_values = {str(expected_value)}
            if observed_values and observed_values != expected_values:
                raise RuntimeError(
                    f"{run_id}: results.csv {column}={sorted(observed_values)!r}, "
                    f"expected {expected_value!r}"
                )
        frame[column] = expected_value
    remaining = [column for column in frame.columns if column not in IDENTITY_COLUMNS]
    return frame[[*IDENTITY_COLUMNS, *remaining]]


def _read_csv_artifacts(
    candidates: dict[str, tuple[Path, dict]], filename: str
) -> pd.DataFrame:
    frames = []
    for run_id, (source, manifest) in sorted(candidates.items()):
        path = source / filename
        if not path.exists():
            raise RuntimeError(f"{run_id}: missing required artifact {filename}")
        frames.append(_attach_identity(pd.read_csv(path), run_id, manifest))
    return pd.concat(frames, ignore_index=True, sort=False)


def _mean_std_summary(
    frame: pd.DataFrame, group_columns: list[str], excluded: set[str] | None = None
) -> pd.DataFrame:
    excluded = excluded or set()
    metrics = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column != "seed" and column not in excluded
    ]
    if not metrics:
        raise RuntimeError("No numeric metrics are available for aggregation")
    summary = frame.groupby(group_columns)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    counts = (
        frame.groupby(group_columns)["seed"]
        .nunique()
        .rename("n_seeds")
        .reset_index()
    )
    return counts.merge(summary, on=group_columns, validate="one_to_one")


def _load_expected_candidates(
    input_root: Path, expected: set[str]
) -> dict[str, tuple[Path, dict]]:
    candidates: dict[str, tuple[Path, dict]] = {}
    for manifest_path in input_root.rglob("run_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = str(manifest.get("run_id", "")).strip()
        if (
            run_id not in expected
            or manifest.get("status") != "complete"
            or not (manifest_path.parent / "results.csv").exists()
        ):
            continue
        previous = candidates.get(run_id)
        if previous:
            identity = ("dataset_id", "dataset_fingerprint", "seed", "backbone", "git_commit")
            if any(previous[1].get(key) != manifest.get(key) for key in identity):
                raise RuntimeError(f"Conflicting exports found for run_id={run_id}")
            if str(manifest.get("finished_at_utc", "")) <= str(
                previous[1].get("finished_at_utc", "")
            ):
                continue
        candidates[run_id] = (manifest_path.parent, manifest)
    return candidates


def bundle(
    input_root: Path,
    output_dir: Path,
    backbone: str,
    seeds: set[int],
    expected: set[str],
) -> Path:
    if not expected:
        raise RuntimeError("Expected run IDs cannot be empty")
    candidates = _load_expected_candidates(input_root.resolve(), expected)
    found = set(candidates)
    if found != expected:
        raise RuntimeError(
            f"Run mismatch for {backbone}: missing={sorted(expected - found)}"
        )

    for run_id, (_, manifest) in candidates.items():
        if str(manifest.get("backbone")) != backbone:
            raise RuntimeError(
                f"{run_id}: backbone={manifest.get('backbone')!r}, expected {backbone!r}"
            )
        if int(manifest.get("seed")) not in seeds:
            raise RuntimeError(
                f"{run_id}: seed={manifest.get('seed')!r} is outside {sorted(seeds)}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    index_rows = []
    result_frames = []
    for run_id, (source, manifest) in sorted(candidates.items()):
        destination = runs_dir / run_id
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        index_rows.append(
            {
                "run_id": run_id,
                "dataset_id": manifest["dataset_id"],
                "dataset_fingerprint": manifest["dataset_fingerprint"],
                "seed": int(manifest["seed"]),
                "backbone": backbone,
                "account_id": manifest.get("account_id", "unknown"),
                "git_commit": manifest.get("git_commit", "unknown"),
                "status": manifest["status"],
            }
        )
        frame = pd.read_csv(source / "results.csv")
        if "method" not in frame.columns:
            raise RuntimeError(f"{run_id}: results.csv has no method column")
        frame = _attach_identity(frame, run_id, manifest)
        result_frames.append(frame)

    index_path = output_dir / "model_run_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    all_results = (
        pd.concat(result_frames, ignore_index=True)
        .drop_duplicates(["dataset_id", "seed", "backbone", "method"], keep="last")
        .sort_values(["dataset_id", "seed", "method"])
        .reset_index(drop=True)
    )
    for dataset_id, dataset_results in all_results.groupby("dataset_id"):
        for seed in sorted(seeds):
            present = set(dataset_results.loc[dataset_results["seed"] == seed, "method"])
            missing = EXPECTED_METHODS - present
            unexpected = present - EXPECTED_METHODS
            if missing or unexpected:
                raise RuntimeError(
                    f"{dataset_id} seed {seed}: missing methods={sorted(missing)}, "
                    f"unexpected methods={sorted(unexpected)}"
                )

    all_results.to_csv(output_dir / "model_all_results.csv", index=False)
    all_results.to_csv(output_dir / "all_seed_results.csv", index=False)
    summary = _mean_std_summary(all_results, ["dataset_id", "backbone", "method"])
    bad_counts = summary[summary["n_seeds"] != len(seeds)]
    if not bad_counts.empty:
        details = bad_counts[["dataset_id", "method", "n_seeds"]].to_dict("records")
        raise RuntimeError(
            f"Three-seed aggregation is incomplete for {backbone}: {details}"
        )
    summary.to_csv(output_dir / "model_three_seed_summary.csv", index=False)
    summary.to_csv(output_dir / "summary_mean_std.csv", index=False)

    official_frames = []
    for dataset_id, dataset_results in all_results.groupby("dataset_id"):
        official = build_official_table(dataset_results, seeds)
        official["dataset_id"] = dataset_id
        official = official[["dataset_id", *[c for c in official if c != "dataset_id"]]]
        official_frames.append(official)
    official_comparison = pd.concat(official_frames, ignore_index=True)
    official_comparison.to_csv(
        output_dir / "official_experiment_comparison.csv", index=False
    )

    paired_rows = []
    bayesian_rows = []
    paired_metrics = [
        metric for metric in ("test_accuracy", "test_f1_macro") if metric in all_results
    ]
    for dataset_id, dataset_results in all_results.groupby("dataset_id"):
        for metric in paired_metrics:
            wide = dataset_results.pivot(index="seed", columns="method", values=metric)
            for method in sorted(EXPECTED_METHODS - {"cnn_baseline"}):
                delta = wide[method] - wide["cnn_baseline"]
                paired_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "method": method,
                        "metric": metric,
                        "delta_vs_cnn_mean": delta.mean(),
                        "delta_vs_cnn_sample_std": delta.std(ddof=1),
                        "n_seeds": len(delta),
                    }
                )
            for reference in sorted(EXPECTED_METHODS - {"gflownet_db_bayesian"}):
                delta = wide["gflownet_db_bayesian"] - wide[reference]
                bayesian_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "comparison": f"gflownet_db_bayesian - {reference}",
                        "metric": metric,
                        "delta_mean": delta.mean(),
                        "delta_sample_std": delta.std(ddof=1),
                        "n_seeds": len(delta),
                    }
                )
    pd.DataFrame(paired_rows).to_csv(output_dir / "paired_delta_vs_cnn.csv", index=False)
    pd.DataFrame(bayesian_rows).to_csv(
        output_dir / "bayesian_vs_core_methods.csv", index=False
    )

    fairness = _read_csv_artifacts(candidates, "fairness.csv").drop_duplicates(
        ["dataset_id", "seed", "backbone"], keep="last"
    )
    rule_columns = ["gflownet", "random", "topk_confidence", "greedy_coverage"]
    missing_fairness_columns = set(rule_columns) - set(fairness.columns)
    if missing_fairness_columns:
        raise RuntimeError(
            f"fairness.csv is missing columns {sorted(missing_fairness_columns)}"
        )
    fairness["verified"] = fairness[rule_columns].nunique(axis=1).eq(1)
    if not fairness["verified"].all():
        raise RuntimeError("Matched-budget verification failed")
    fairness.to_csv(output_dir / "matched_budget_audit.csv", index=False)

    rule_quality = _read_csv_artifacts(candidates, "rule_set_quality.csv").drop_duplicates(
        ["dataset_id", "seed", "backbone", "method"], keep="last"
    )
    for dataset_id, dataset_quality in rule_quality.groupby("dataset_id"):
        for seed in sorted(seeds):
            present = set(dataset_quality.loc[dataset_quality["seed"] == seed, "method"])
            missing = RULE_QUALITY_METHODS - present
            if missing:
                raise RuntimeError(
                    f"{dataset_id} seed {seed}: missing rule-quality methods {sorted(missing)}"
                )
    quality_summary = _mean_std_summary(
        rule_quality, ["dataset_id", "backbone", "method"]
    )
    rule_quality.to_csv(output_dir / "rule_set_quality_all_seeds.csv", index=False)
    quality_summary.to_csv(output_dir / "rule_set_quality_mean_std.csv", index=False)

    ranking = _read_csv_artifacts(candidates, "rule_ranking_metrics.csv").drop_duplicates(
        ["dataset_id", "seed", "backbone"], keep="last"
    )
    ranking_summary = _mean_std_summary(ranking, ["dataset_id", "backbone"])
    ranking.to_csv(output_dir / "rule_ranking_metrics_all_seeds.csv", index=False)
    ranking_summary.to_csv(output_dir / "rule_ranking_metrics_mean_std.csv", index=False)

    exact_metrics = {}
    for run_id, (source, manifest) in sorted(candidates.items()):
        exact_path = source / "exact_test_metrics.json"
        if not exact_path.exists():
            raise RuntimeError(f"{run_id}: missing required artifact exact_test_metrics.json")
        payload = json.loads(exact_path.read_text(encoding="utf-8"))
        for item in payload:
            item = dict(item)
            item["dataset_id"] = manifest["dataset_id"]
            item["seed"] = int(manifest["seed"])
            item["backbone"] = backbone
            key = (item["dataset_id"], item["seed"], str(item.get("method", "")))
            exact_metrics[key] = item
    (output_dir / "exact_test_metrics_all_seeds.json").write_text(
        json.dumps(list(exact_metrics.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    runtime_paths = [source / "runtime_summary.csv" for source, _ in candidates.values()]
    if all(path.exists() for path in runtime_paths):
        runtime = _read_csv_artifacts(candidates, "runtime_summary.csv").drop_duplicates(
            ["dataset_id", "seed", "backbone", "stage"], keep="last"
        )
        measured = runtime[
            runtime.get("runtime_status", pd.Series(index=runtime.index, dtype=str))
            == "measured"
        ]
        runtime.to_csv(output_dir / "runtime_all_seeds.csv", index=False)
        if not measured.empty:
            _mean_std_summary(
                measured, ["dataset_id", "backbone", "stage"]
            ).to_csv(output_dir / "runtime_mean_std.csv", index=False)

    bundle_manifest = {
        "schema_version": 1,
        "bundle_type": "kaggle_model_bundle",
        "backbone": backbone,
        "seeds": sorted(seeds),
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(index_rows),
        "run_ids": [row["run_id"] for row in index_rows],
        "git_commits": sorted({row["git_commit"] for row in index_rows}),
        "result_row_count": len(all_results),
        "summary_row_count": len(summary),
        "summary_file": "model_three_seed_summary.csv",
        "official_comparison_file": "official_experiment_comparison.csv",
    }
    (output_dir / "model_bundle_manifest.json").write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    archive_path = output_dir.parent / f"{output_dir.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_dir, arcname=output_dir.name)
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/model_bundle"))
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--expected-run-ids", nargs="+", required=True)
    args = parser.parse_args()
    archive = bundle(
        args.input_root,
        args.output_dir,
        args.backbone,
        set(args.seeds),
        set(args.expected_run_ids),
    )
    print(f"Model bundle ready: {args.output_dir}")
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
