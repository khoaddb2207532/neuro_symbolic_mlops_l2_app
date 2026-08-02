"""Sinh notebook launcher từ experiment registry."""

import argparse
import csv
import json
from pathlib import Path


def generate(registry_path: Path, template_path: Path, output_dir: Path, git_commit: str) -> None:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with registry_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        if "REPLACE_" in row["data_dir"]:
            raise ValueError(
                f"Run {row['run_id']} chưa có data_dir thật: {row['data_dir']}"
            )
        notebook = json.loads(json.dumps(template, ensure_ascii=False))
        parameter_cell = next(
            cell
            for cell in notebook["cells"]
            if "experiment-parameters" in cell.get("metadata", {}).get("tags", [])
        )
        parameter_cell["source"] = [
            f'DATASET_ID = {row["dataset_id"]!r}\n',
            f'DATA_DIR = {row["data_dir"]!r}\n',
            f'SEED = {int(row["seed"])}\n',
            f'BACKBONE = {row["backbone"]!r}\n',
            f'ACCOUNT_ID = {row["account_id"]!r}\n',
            f'RUN_ID = {row["run_id"]!r}\n',
            f'GIT_COMMIT = {git_commit!r}',
        ]
        destination = output_dir / f'{row["run_id"]}.ipynb'
        destination.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="experiments/experiment_registry.csv")
    parser.add_argument("--template", default="managed-experiment-runner-template.ipynb")
    parser.add_argument("--output-dir", default="generated_kaggle_notebooks")
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    generate(
        Path(args.registry),
        Path(args.template),
        Path(args.output_dir),
        args.git_commit,
    )
