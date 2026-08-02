import json
from pathlib import Path

from pipelines.run_core_seed_experiment import (
    _dataset_fingerprint,
    _find_restorable_output,
)


def _managed_run(root: Path, run_id: str, dataset_id: str, backbone: str) -> Path:
    output = root / run_id
    (output / "02_features").mkdir(parents=True)
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "seed": 42,
                "backbone": backbone,
                "status": "failed",
            }
        ),
        encoding="utf-8",
    )
    return output


def test_resume_selects_exact_dataset_backbone_and_run_id(tmp_path: Path) -> None:
    _managed_run(tmp_path, "culture-a__alexnet__db__seed_42", "culture-a", "alexnet")
    expected = _managed_run(
        tmp_path,
        "culture-b__resnet50__db__seed_42",
        "culture-b",
        "resnet50",
    )
    found = _find_restorable_output(
        tmp_path,
        42,
        "resnet50",
        "culture-b",
        "culture-b__resnet50__db__seed_42",
    )
    assert found == ("directory", expected)


def test_dataset_fingerprint_ignores_other_working_files(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        folder = tmp_path / split / "class-a"
        folder.mkdir(parents=True)
        (folder / "image.bin").write_bytes(b"image")
    first = _dataset_fingerprint(tmp_path)
    (tmp_path / "repository-output.bin").write_bytes(b"changes between resumes")
    second = _dataset_fingerprint(tmp_path)
    assert first == second
