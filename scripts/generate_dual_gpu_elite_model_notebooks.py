"""Generate one dual-GPU notebook per backbone, containing all seeds."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# from scripts.generate_dual_gpu_elite_notebooks import SUPPORTED_BACKBONES

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

def generate(
    registry_path: Path,
    template_path: Path,
    output_dir: Path,
    git_commit: str,
    *,
    backbones: list[str],
    seeds: list[int],
    datasets: list[str],
) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", git_commit):
        raise ValueError("--git-commit phải là full 40-character SHA đã push.")
    unknown_backbones = set(backbones) - SUPPORTED_BACKBONES
    if unknown_backbones:
        raise ValueError(f"Backbone không hỗ trợ: {sorted(unknown_backbones)}")

    with registry_path.open(newline="", encoding="utf-8") as file:
        registry_rows = list(csv.DictReader(file))
    dataset_dirs = {}
    for row in registry_rows:
        dataset_dirs.setdefault(row["dataset_id"], row["data_dir"])
    unknown_datasets = set(datasets) - set(dataset_dirs)
    if unknown_datasets:
        raise ValueError(f"Dataset chưa có trong registry: {sorted(unknown_datasets)}")

    template = json.loads(template_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for backbone in backbones:
        runs = [
            {
                "dataset_id": dataset_id,
                "data_dir": dataset_dirs[dataset_id],
                "seed": seed,
                "prior_run_id": (
                    f"{dataset_id}__{backbone}__db__seed_{seed}"
                ),
            }
            for dataset_id in datasets
            for seed in seeds
        ]
        notebook = json.loads(json.dumps(template, ensure_ascii=False))
        parameter_cell = next(
            cell
            for cell in notebook["cells"]
            if "model-all-seeds-parameters"
            in cell.get("metadata", {}).get("tags", [])
        )
        parameter_cell["source"] = [
            f"BACKBONE = {backbone!r}\n",
            f"GIT_COMMIT = {git_commit.lower()!r}\n",
            f"SEEDS = {seeds!r}\n",
            f"RUNS = {runs!r}\n",
            "MC_SAMPLES = 32\n",
            "GFLOWNET_ITERATIONS = 5000\n",
            "NUM_EPOCHS = 100\n",
            "PATIENCE = 5",
        ]
        destination = output_dir / f"dual_elite_all_seeds_{backbone}.ipynb"
        destination.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        generated.append(destination)
        print(destination)

    if len(generated) != len(backbones):
        raise RuntimeError(
            f"Sinh thiếu notebook: {len(generated)}/{len(backbones)}"
        )
    print(
        f"Generated {len(generated)} model notebooks; each contains "
        f"{len(datasets)} datasets x {len(seeds)} seeds."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="experiments/experiment_registry.csv")
    parser.add_argument(
        "--template",
        default="dual-gpu-elite-model-all-seeds-template.ipynb",
    )
    parser.add_argument(
        "--output-dir",
        default="generated_dual_gpu_elite_model_notebooks",
    )
    parser.add_argument("--git-commit", required=True)
    parser.add_argument(
        "--backbones",
        nargs="+",
        required=True,
        choices=sorted(SUPPORTED_BACKBONES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
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
