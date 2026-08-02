"""Collect lightweight managed run exports into one Kaggle account bundle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def _load_candidates(input_root: Path, account_id: str) -> dict[str, tuple[Path, dict]]:
    candidates: dict[str, tuple[Path, dict]] = {}
    for manifest_path in input_root.rglob("run_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = str(manifest.get("run_id", "")).strip()
        if (
            not run_id
            or manifest.get("status") != "complete"
            or manifest.get("account_id") != account_id
            or not (manifest_path.parent / "results.csv").exists()
        ):
            continue
        previous = candidates.get(run_id)
        if previous:
            previous_manifest = previous[1]
            identity = ("dataset_id", "dataset_fingerprint", "seed", "backbone", "git_commit")
            if any(previous_manifest.get(key) != manifest.get(key) for key in identity):
                raise RuntimeError(f"Conflicting exports found for run_id={run_id}")
            if str(manifest.get("finished_at_utc", "")) <= str(previous_manifest.get("finished_at_utc", "")):
                continue
        candidates[run_id] = (manifest_path.parent, manifest)
    return candidates


def bundle(input_root: Path, output_dir: Path, account_id: str, expected: set[str]) -> Path:
    candidates = _load_candidates(input_root.resolve(), account_id)
    found = set(candidates)
    if expected and found != expected:
        raise RuntimeError(
            f"Run mismatch for {account_id}: missing={sorted(expected - found)}, "
            f"unexpected={sorted(found - expected)}"
        )
    if not candidates:
        raise RuntimeError(f"No completed managed exports found for account_id={account_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    index_rows = []
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
                "seed": manifest["seed"],
                "backbone": manifest["backbone"],
                "account_id": account_id,
                "git_commit": manifest.get("git_commit", "unknown"),
                "status": manifest["status"],
            }
        )

    index_path = output_dir / "account_run_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    bundle_manifest = {
        "schema_version": 1,
        "bundle_type": "kaggle_account_bundle",
        "account_id": account_id,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(index_rows),
        "run_ids": [row["run_id"] for row in index_rows],
        "git_commits": sorted({row["git_commit"] for row in index_rows}),
    }
    (output_dir / "account_bundle_manifest.json").write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    archive_path = output_dir.parent / f"{output_dir.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_dir, arcname=output_dir.name)
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/account_bundle"))
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--expected-run-ids", nargs="*", default=[])
    args = parser.parse_args()
    archive = bundle(args.input_root, args.output_dir, args.account_id, set(args.expected_run_ids))
    print(f"Account bundle ready: {args.output_dir}")
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
