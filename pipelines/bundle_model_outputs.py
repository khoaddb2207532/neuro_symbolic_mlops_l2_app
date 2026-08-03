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


IDENTITY_COLUMNS = ("run_id", "dataset_id", "seed", "backbone")


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

    all_results = pd.concat(result_frames, ignore_index=True)
    all_results.to_csv(output_dir / "model_all_results.csv", index=False)
    metric_columns = [
        column
        for column in all_results.select_dtypes(include="number").columns
        if column != "seed"
    ]
    if not metric_columns:
        raise RuntimeError("results.csv has no numeric metric columns to aggregate")
    summary = (
        all_results.groupby(["dataset_id", "method"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    seed_counts = (
        all_results.groupby(["dataset_id", "method"])["seed"]
        .nunique()
        .rename("n_seeds")
        .reset_index()
    )
    summary = seed_counts.merge(summary, on=["dataset_id", "method"], validate="one_to_one")
    bad_counts = summary[summary["n_seeds"] != len(seeds)]
    if not bad_counts.empty:
        details = bad_counts[["dataset_id", "method", "n_seeds"]].to_dict("records")
        raise RuntimeError(
            f"Three-seed aggregation is incomplete for {backbone}: {details}"
        )
    summary.to_csv(output_dir / "model_three_seed_summary.csv", index=False)

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
