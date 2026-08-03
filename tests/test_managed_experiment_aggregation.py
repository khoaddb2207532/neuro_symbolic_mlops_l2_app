import json
from pathlib import Path

import pandas as pd

from pipelines.aggregate_multi_dataset_results import EXPECTED_METHODS, aggregate
from pipelines.bundle_account_outputs import bundle


def _create_run(root: Path, row: dict) -> None:
    folder = root / row["account_id"] / row["run_id"]
    folder.mkdir(parents=True)
    manifest = {
        **row,
        "dataset_fingerprint": f"fingerprint-{row['dataset_id']}",
        "git_commit": "abc123",
        "status": "complete",
    }
    (folder / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame(
        {
            "method": sorted(EXPECTED_METHODS),
            "test_accuracy": [0.70 + index / 100 for index in range(6)],
            "test_f1_macro": [0.60 + index / 100 for index in range(6)],
        }
    ).to_csv(folder / "results.csv", index=False)


def test_bundle_and_aggregate_complete_matrix(tmp_path: Path) -> None:
    rows = []
    for dataset_index, dataset_id in enumerate(("dataset-a", "dataset-b")):
        for seed_index, seed in enumerate((46, 48, 50)):
            account_id = f"account-{dataset_index * 2 + (1 if seed_index < 1 else 2)}"
            row = {
                "run_id": f"{dataset_id}__mobilenetv3_small__db__seed_{seed}",
                "dataset_id": dataset_id,
                "seed": seed,
                "backbone": "mobilenetv3_small",
                "account_id": account_id,
            }
            rows.append(row)
            _create_run(tmp_path / "inputs", row)
    registry = tmp_path / "registry.csv"
    pd.DataFrame(rows).to_csv(registry, index=False)

    account_rows = [row for row in rows if row["account_id"] == "account-1"]
    archive = bundle(
        tmp_path / "inputs",
        tmp_path / "bundle_account-1",
        "account-1",
        {row["run_id"] for row in account_rows},
    )
    assert archive.exists()
    assert len(pd.read_csv(tmp_path / "bundle_account-1" / "account_run_index.csv")) == 1

    output = tmp_path / "summary"
    aggregate(tmp_path / "inputs", registry, output)
    assert len(pd.read_csv(output / "multi_dataset_run_audit.csv")) == 6
    assert len(pd.read_csv(output / "multi_dataset_all_results.csv")) == 36
    assert len(pd.read_csv(output / "dataset_method_summary.csv")) == 12
