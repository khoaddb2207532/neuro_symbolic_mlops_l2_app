import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines.bundle_model_outputs import bundle
from scripts.generate_kaggle_bundle_notebooks import generate


def _write_run(root: Path, run_id: str, backbone: str, seed: int) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "dataset_id": "culture-a",
        "dataset_fingerprint": "fingerprint-a",
        "seed": seed,
        "backbone": backbone,
        "account_id": "account-1",
        "git_commit": "abc123",
        "status": "complete",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "results.csv").write_text("method,test_accuracy\ncnn_baseline,0.8\n", encoding="utf-8")


def test_generate_one_bundle_per_model_with_three_seeds(tmp_path: Path) -> None:
    rows = []
    for backbone in ("alexnet", "resnet50"):
        for dataset in ("culture-a", "culture-b"):
            for seed in (42, 46, 48, 50):
                rows.append(
                    {
                        "run_id": f"{dataset}__{backbone}__db__seed_{seed}",
                        "dataset_id": dataset,
                        "seed": seed,
                        "backbone": backbone,
                    }
                )
    registry = tmp_path / "registry.csv"
    pd.DataFrame(rows).to_csv(registry, index=False)

    generated = generate(
        registry,
        Path("managed-model-bundle-template.ipynb"),
        tmp_path / "generated",
        "abc123",
    )

    assert [path.name for path in generated] == ["bundle_alexnet.ipynb", "bundle_resnet50.ipynb"]
    for path in generated:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        parameter_cell = next(
            cell for cell in notebook["cells"]
            if "bundle-parameters" in cell.get("metadata", {}).get("tags", [])
        )
        source = "".join(parameter_cell["source"])
        assert "SEEDS = [46, 48, 50]" in source
        assert "seed_42" not in source
        assert source.count("__db__seed_") == 6


def test_generate_rejects_model_missing_requested_seed(tmp_path: Path) -> None:
    registry = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {"run_id": f"run-{seed}", "seed": seed, "backbone": "alexnet"}
            for seed in (46, 48)
        ]
    ).to_csv(registry, index=False)

    with pytest.raises(ValueError, match=r"missing requested seeds: \[50\]"):
        generate(
            registry,
            Path("managed-model-bundle-template.ipynb"),
            tmp_path / "generated",
            "abc123",
        )


def test_generate_expands_registry_matrix_for_requested_backbones(tmp_path: Path) -> None:
    registry = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {
                "run_id": f"culture-a__vit_b_32__db__seed_{seed}",
                "dataset_id": "culture-a",
                "loss_type": "db",
                "seed": seed,
                "backbone": "vit_b_32",
            }
            for seed in (46, 48, 50)
        ]
    ).to_csv(registry, index=False)

    generated = generate(
        registry,
        Path("managed-model-bundle-template.ipynb"),
        tmp_path / "generated",
        "abc123",
        backbones=("alexnet", "vit_b_32"),
    )

    assert [path.name for path in generated] == ["bundle_alexnet.ipynb", "bundle_vit_b_32.ipynb"]
    alexnet = json.loads(generated[0].read_text(encoding="utf-8"))
    source = "".join(
        next(
            cell for cell in alexnet["cells"]
            if "bundle-parameters" in cell.get("metadata", {}).get("tags", [])
        )["source"]
    )
    assert "culture-a__alexnet__db__seed_46" in source
    assert "vit_b_32" not in source


def test_model_bundle_accepts_runs_from_multiple_accounts(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    expected = set()
    for index, seed in enumerate((46, 48, 50), start=1):
        run_id = f"culture-a__resnet50__db__seed_{seed}"
        _write_run(input_root / f"account-{index}", run_id, "resnet50", seed)
        expected.add(run_id)

    archive = bundle(
        input_root,
        tmp_path / "bundle_resnet50",
        "resnet50",
        {46, 48, 50},
        expected,
    )

    assert archive.exists()
    index = pd.read_csv(tmp_path / "bundle_resnet50" / "model_run_index.csv")
    assert set(index["seed"]) == {46, 48, 50}
    manifest = json.loads(
        (tmp_path / "bundle_resnet50" / "model_bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["backbone"] == "resnet50"
    assert manifest["seeds"] == [46, 48, 50]
