"""Generate one dual-GPU TB/DB Bayesian-Elite Kaggle notebook per seed."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _dataset_setup(dataset_id: str, data_dir: str) -> list[str]:
    if dataset_id == "culture-b":
        return [
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "source = Path('/kaggle/input/datasets/utkarshsaxenadn/fast-food-classification-dataset/Fast Food Classification V2')\n",
            "mapping = {'train': 'Train', 'test': 'Test', 'val': 'Valid'}\n",
            "for destination, origin in mapping.items():\n",
            "    path = Path('/kaggle/working') / destination\n",
            "    if path.is_symlink() or path.exists():\n",
            "        path.unlink()\n",
            "    path.symlink_to(source / origin, target_is_directory=True)\n",
            "assert Path(DATA_DIR).exists(), DATA_DIR\n",
        ]
    return [
        "from pathlib import Path\n",
        "assert Path(DATA_DIR).exists(), f'DATA_DIR không tồn tại: {DATA_DIR}'\n",
    ]


def generate(
    registry_path: Path,
    template_path: Path,
    output_dir: Path,
    git_commit: str,
) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", git_commit):
        raise ValueError("--git-commit phải là full 40-character commit SHA đã push.")
    template = json.loads(template_path.read_text(encoding="utf-8"))
    with registry_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    seen = set()
    for row in rows:
        # One notebook launches both objectives; DB registry rows are the
        # canonical prior-stage identity and prevent duplicate TB/DB launchers.
        if row.get("loss_type", "db") != "db":
            continue
        identity = (row["dataset_id"], row["backbone"], int(row["seed"]))
        if identity in seen:
            continue
        seen.add(identity)
        if "REPLACE_" in row["data_dir"]:
            raise ValueError(f"Run {row['run_id']} chưa có data_dir thật.")

        notebook = json.loads(json.dumps(template, ensure_ascii=False))
        parameter_cell = next(
            cell
            for cell in notebook["cells"]
            if "dual-elite-parameters" in cell.get("metadata", {}).get("tags", [])
        )
        parameter_cell["source"] = [
            f"DATASET_ID = {row['dataset_id']!r}\n",
            f"DATA_DIR = {row['data_dir']!r}\n",
            f"SEED = {int(row['seed'])}\n",
            f"BACKBONE = {row['backbone']!r}\n",
            f"PRIOR_RUN_ID = {row['run_id']!r}\n",
            f"GIT_COMMIT = {git_commit.lower()!r}\n",
            "MC_SAMPLES = 32\n",
            "GFLOWNET_ITERATIONS = 5000\n",
            "NUM_EPOCHS = 100\n",
            "PATIENCE = 5",
        ]
        setup_cell = next(
            cell
            for cell in notebook["cells"]
            if "dataset-setup" in cell.get("metadata", {}).get("tags", [])
        )
        setup_cell["source"] = _dataset_setup(row["dataset_id"], row["data_dir"])

        destination_dir = output_dir / row["dataset_id"] / row["backbone"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"dual_elite_seed_{int(row['seed'])}.ipynb"
        destination.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="experiments/experiment_registry.csv")
    parser.add_argument("--template", default="dual-gpu-elite-seed-template.ipynb")
    parser.add_argument("--output-dir", default="generated_dual_gpu_elite_notebooks")
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    generate(
        Path(args.registry),
        Path(args.template),
        Path(args.output_dir),
        args.git_commit,
    )
