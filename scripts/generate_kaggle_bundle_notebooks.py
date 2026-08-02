"""Sinh một bundle notebook cho mỗi account_id trong registry."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="experiments/experiment_registry.csv")
    parser.add_argument("--template", default="managed-account-bundle-template.ipynb")
    parser.add_argument("--output-dir", default="generated_kaggle_notebooks")
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()

    registry = pd.read_csv(args.registry)
    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for account_id, rows in registry.groupby("account_id", sort=True):
        notebook = json.loads(json.dumps(template, ensure_ascii=False))
        cell = next(
            item for item in notebook["cells"]
            if "bundle-parameters" in item.get("metadata", {}).get("tags", [])
        )
        run_ids = rows["run_id"].tolist()
        cell["source"] = [
            f"ACCOUNT_ID = {account_id!r}\n",
            f"EXPECTED_RUN_IDS = {run_ids!r}\n",
            f"GIT_COMMIT = {args.git_commit!r}",
        ]
        destination = output_dir / f"bundle_{account_id}.ipynb"
        destination.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(destination)


if __name__ == "__main__":
    main()
