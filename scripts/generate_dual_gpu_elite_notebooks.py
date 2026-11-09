"""Generate one dual-GPU TB/DB Bayesian-Elite Kaggle notebook per seed."""

from __future__ import annotations

import argparse
import csv
import json
import re
from itertools import product
from pathlib import Path


SUPPORTED_BACKBONES = {
    "mobilenetv3_small",
    "shufflenet_v2_x1_0",
    "alexnet",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "swin_t",
    "vit_b_16",
    "vit_b_32",
}


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
    *,
    backbones: list[str] | None = None,
    seeds: list[int] | None = None,
    datasets: list[str] | None = None,
) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", git_commit):
        raise ValueError("--git-commit phải là full 40-character commit SHA đã push.")
    template = json.loads(template_path.read_text(encoding="utf-8"))
    with registry_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    dataset_config = {}
    for row in rows:
        dataset_config.setdefault(row["dataset_id"], row["data_dir"])
    selected_datasets = datasets or sorted(dataset_config)
    unknown_datasets = set(selected_datasets) - set(dataset_config)
    if unknown_datasets:
        raise ValueError(f"Dataset chưa có data_dir trong registry: {sorted(unknown_datasets)}")
    selected_backbones = backbones or sorted({row["backbone"] for row in rows})
    unknown_backbones = set(selected_backbones) - SUPPORTED_BACKBONES
    if unknown_backbones:
        raise ValueError(f"Backbone không được hỗ trợ: {sorted(unknown_backbones)}")
    selected_seeds = seeds or sorted({int(row["seed"]) for row in rows})

    generated_count = 0
    for dataset_id, backbone, seed in product(
        selected_datasets, selected_backbones, selected_seeds
    ):
        data_dir = dataset_config[dataset_id]
        run_id = f"{dataset_id}__{backbone}__db__seed_{seed}"
        row = {
            "dataset_id": dataset_id,
            "data_dir": data_dir,
            "seed": seed,
            "backbone": backbone,
            "run_id": run_id,
        }
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
        generated_count += 1
    expected = len(selected_datasets) * len(selected_backbones) * len(selected_seeds)
    if generated_count != expected:
        raise RuntimeError(f"Sinh thiếu notebook: {generated_count}/{expected}")
    print(
        f"Generated {generated_count} notebooks = "
        f"{len(selected_datasets)} datasets x {len(selected_backbones)} backbones "
        f"x {len(selected_seeds)} seeds."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="experiments/experiment_registry.csv")
    parser.add_argument("--template", default="dual-gpu-elite-seed-template.ipynb")
    parser.add_argument("--output-dir", default="generated_dual_gpu_elite_notebooks")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--backbones", nargs="+", choices=sorted(SUPPORTED_BACKBONES))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()
    generate(
        Path(args.registry),
        Path(args.template),
        Path(args.output_dir),
        args.git_commit,
        backbones=args.backbones,
        seeds=args.seeds,
        datasets=args.datasets,
    )
