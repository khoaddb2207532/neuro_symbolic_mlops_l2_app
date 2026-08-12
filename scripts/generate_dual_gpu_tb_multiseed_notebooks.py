"""Generate one two-GPU, multi-seed GFlowNet-TB notebook per backbone."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SUPPORTED_BACKBONES = {
    "mobilenetv3_small",
    "alexnet",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "swin_t",
    "vit_b_16",
    "vit_b_32",
}


DEFAULT_BACKBONES = (
    "mobilenetv3_small",
    "alexnet",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "vit_b_32",
    "swin_t",
)


def generate(
    registry_path: Path,
    template_path: Path,
    output_dir: Path,
    *,
    git_ref: str,
    backbones: list[str] | tuple[str, ...] = DEFAULT_BACKBONES,
    seeds: list[int] | tuple[int, ...] = (42, 44, 46, 48, 50),
    datasets: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    if not git_ref.strip():
        raise ValueError("--git-ref không được rỗng.")
    unknown = set(backbones) - set(SUPPORTED_BACKBONES)
    if unknown:
        raise ValueError(f"Backbone không hỗ trợ: {sorted(unknown)}")
    with registry_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    data_dirs: dict[str, str] = {}
    for row in rows:
        data_dirs.setdefault(row["dataset_id"], row["data_dir"])
    selected_datasets = list(datasets) if datasets else sorted(data_dirs)
    missing = set(selected_datasets) - set(data_dirs)
    if missing:
        raise ValueError(f"Dataset chưa có trong registry: {sorted(missing)}")
    template = json.loads(template_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for backbone in backbones:
        runs = [
            {
                "dataset_id": dataset_id,
                "data_dir": data_dirs[dataset_id],
                "seed": int(seed),
                "prior_run_id": f"{dataset_id}__{backbone}__db__seed_{int(seed)}",
            }
            for dataset_id in selected_datasets
            for seed in seeds
        ]
        notebook = json.loads(json.dumps(template, ensure_ascii=False))
        cell = next(
            item
            for item in notebook["cells"]
            if "tb-multiseed-parameters" in item.get("metadata", {}).get("tags", [])
        )
        cell["source"] = [
            f"BACKBONE = {backbone!r}\n",
            f"GIT_REF = {git_ref!r}\n",
            f"SEEDS = {list(map(int, seeds))!r}\n",
            f"RUNS = {runs!r}\n",
            "MC_SAMPLES = 32\n",
            "GFLOWNET_ITERATIONS = 5000\n",
            "NUM_EPOCHS = 100\n",
            "PATIENCE = 5",
        ]
        destination = output_dir / f"dual_gpu_tb_multiseed_{backbone}.ipynb"
        destination.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        generated.append(destination)
        print(destination)
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="experiments/experiment_registry.csv")
    parser.add_argument(
        "--template", default="dual-gpu-tb-multiseed-template.ipynb"
    )
    parser.add_argument(
        "--output-dir", default="generated_dual_gpu_tb_multiseed_notebooks"
    )
    parser.add_argument("--git-ref", default="main")
    parser.add_argument(
        "--backbones", nargs="+", choices=sorted(SUPPORTED_BACKBONES), default=list(DEFAULT_BACKBONES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 44, 46, 48, 50])
    parser.add_argument("--datasets", nargs="+")
    arguments = parser.parse_args()
    generate(
        Path(arguments.registry),
        Path(arguments.template),
        Path(arguments.output_dir),
        git_ref=arguments.git_ref,
        backbones=arguments.backbones,
        seeds=arguments.seeds,
        datasets=arguments.datasets,
    )
