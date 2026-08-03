"""Generate one Kaggle bundle notebook per model in the experiment registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_SEEDS = (46, 48, 50)


def generate(
    registry_path: Path,
    template_path: Path,
    output_dir: Path,
    git_commit: str,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    backbones: Iterable[str] | None = None,
) -> list[Path]:
    selected_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not selected_seeds:
        raise ValueError("At least one seed is required")

    registry = pd.read_csv(registry_path, dtype={"seed": int})
    required = {"run_id", "seed", "backbone"}
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"Registry is missing columns: {sorted(missing)}")
    registry = registry[registry["seed"].isin(selected_seeds)]
    if registry.empty:
        raise ValueError(f"Registry has no runs for seeds {list(selected_seeds)}")

    requested_backbones = tuple(dict.fromkeys(str(item) for item in (backbones or ())))
    if requested_backbones:
        expansion_columns = {"dataset_id", "loss_type"}
        missing_expansion_columns = expansion_columns - set(registry.columns)
        if missing_expansion_columns:
            raise ValueError(
                "Registry is missing columns needed for backbone expansion: "
                f"{sorted(missing_expansion_columns)}"
            )
        expanded = []
        for backbone in requested_backbones:
            model_rows = registry.copy()
            model_rows["backbone"] = backbone
            model_rows["run_id"] = model_rows.apply(
                lambda row: (
                    f"{row['dataset_id']}__{backbone}__{row['loss_type']}__seed_{int(row['seed'])}"
                ),
                axis=1,
            )
            expanded.append(model_rows)
        registry = pd.concat(expanded, ignore_index=True)

    template = json.loads(template_path.read_text(encoding="utf-8"))
    destinations: list[Path] = []
    for backbone, rows in registry.groupby("backbone", sort=True):
        present_seeds = set(rows["seed"].astype(int))
        absent_seeds = set(selected_seeds) - present_seeds
        if absent_seeds:
            raise ValueError(
                f"Model {backbone!r} is missing requested seeds: {sorted(absent_seeds)}"
            )

        notebook = json.loads(json.dumps(template, ensure_ascii=False))
        cell = next(
            item
            for item in notebook["cells"]
            if "bundle-parameters" in item.get("metadata", {}).get("tags", [])
        )
        run_ids = rows.sort_values(["seed", "run_id"])["run_id"].tolist()
        cell["source"] = [
            f"BACKBONE = {str(backbone)!r}\n",
            f"SEEDS = {list(selected_seeds)!r}\n",
            f"EXPECTED_RUN_IDS = {run_ids!r}\n",
            f"GIT_COMMIT = {git_commit!r}",
        ]

        model_output_dir = output_dir / str(backbone)
        model_output_dir.mkdir(parents=True, exist_ok=True)
        destination = model_output_dir / f"bundle_{backbone}.ipynb"
        destination.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        destinations.append(destination)
    return destinations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("experiments/experiment_registry.csv"))
    parser.add_argument("--template", type=Path, default=Path("managed-model-bundle-template.ipynb"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated_kaggle_notebooks"))
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--backbones",
        nargs="+",
        help="Optional model list; reuse the registry dataset/seed matrix for each model",
    )
    args = parser.parse_args()

    for destination in generate(
        args.registry,
        args.template,
        args.output_dir,
        args.git_commit,
        args.seeds,
        args.backbones,
    ):
        print(destination)


if __name__ == "__main__":
    main()
