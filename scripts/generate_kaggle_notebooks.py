"""Sinh notebook launcher từ experiment registry."""

import argparse
import csv
import json
from pathlib import Path


DEFAULT_DATASET_DIRS = {
    "culture-a": "/kaggle/input/datasets/dangduykhoab2207532/"
    "vietnamese-cultural-dataset/vietnamese_cultural_dataset",
    "culture-b": "/kaggle/working",
}


def _generate_rows(rows, template_path: Path, output_dir: Path, git_commit: str) -> list[Path]:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
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
        if row["dataset_id"] == "culture-b":
            clone_cell_index = next(
                index
                for index, cell in enumerate(notebook["cells"])
                if cell.get("cell_type") == "code"
                and any("GITHUB_TOKEN" in line for line in cell.get("source", []))
            )
            dataset_b_cells = [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {"tags": ["dataset-b-warning-control"]},
                    "outputs": [],
                    "source": [
                        "import warnings\n",
                        "import os\n",
                        "\n",
                        "# Tắt cảnh báo Python và log không cần thiết.\n",
                        "warnings.filterwarnings(\"ignore\")\n",
                        "os.environ[\"TORCH_CPP_LOG_LEVEL\"] = \"ERROR\"\n",
                        "os.environ[\"PYTHONWARNINGS\"] = \"ignore\"\n",
                        "os.environ[\"WANDB_SILENT\"] = \"true\"",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {"tags": ["dataset-b-symlinks"]},
                    "outputs": [],
                    "source": [
                        "%%bash\n",
                        "set -e\n",
                        "DATA_PATH=\"/kaggle/input/datasets/utkarshsaxenadn/fast-food-classification-dataset/Fast Food Classification V2\"\n",
                        "WORK_PATH=\"/kaggle/working\"\n",
                        "\n",
                        "# -sfn cho phép chạy lại cell an toàn khi resume.\n",
                        "ln -sfn \"$DATA_PATH/Train\" \"$WORK_PATH/train\"\n",
                        "ln -sfn \"$DATA_PATH/Test\" \"$WORK_PATH/test\"\n",
                        "ln -sfn \"$DATA_PATH/Valid\" \"$WORK_PATH/val\"\n",
                        "ls -l \"$WORK_PATH/train\" \"$WORK_PATH/test\" \"$WORK_PATH/val\"",
                    ],
                },
            ]
            notebook["cells"][
                clone_cell_index + 1:clone_cell_index + 1
            ] = dataset_b_cells
        model_output_dir = output_dir / row["backbone"]
        model_output_dir.mkdir(parents=True, exist_ok=True)
        destination = model_output_dir / f'{row["run_id"]}.ipynb'
        destination.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        generated.append(destination)
        print(destination)
    return generated


def generate(registry_path: Path, template_path: Path, output_dir: Path, git_commit: str) -> list[Path]:
    with registry_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return _generate_rows(rows, template_path, output_dir, git_commit)


def generate_matrix(
    template_path: Path,
    output_dir: Path,
    git_commit: str,
    *,
    backbones: list[str],
    seeds: list[int],
    datasets: list[str],
    dataset_dirs: dict[str, str] | None = None,
    account_id: str = "account-1",
) -> list[Path]:
    """Sinh trực tiếp ma trận core-prior, không cần experiment registry CSV."""
    resolved_dirs = {**DEFAULT_DATASET_DIRS, **(dataset_dirs or {})}
    unknown = sorted(set(datasets) - set(resolved_dirs))
    if unknown:
        raise ValueError(f"Thiếu data_dir cho dataset: {unknown}")
    rows = [
        {
            "run_id": f"{dataset}__{backbone}__db__seed_{seed}",
            "dataset_id": dataset,
            "data_dir": resolved_dirs[dataset],
            "seed": seed,
            "backbone": backbone,
            "account_id": account_id,
        }
        for backbone in backbones
        for dataset in datasets
        for seed in seeds
    ]
    generated = _generate_rows(rows, template_path, output_dir, git_commit)
    expected = len(backbones) * len(datasets) * len(seeds)
    if len(generated) != expected:
        raise RuntimeError(f"Sinh thiếu notebook: {len(generated)}/{expected}")
    print(f"Generated {len(generated)} core-prior notebooks without CSV.")
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="experiments/experiment_registry.csv")
    parser.add_argument("--template", default="managed-experiment-runner-template.ipynb")
    parser.add_argument("--output-dir", default="generated_kaggle_notebooks")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--matrix", action="store_true", help="Sinh ma trận trực tiếp, không đọc CSV.")
    parser.add_argument("--backbones", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--account-id", default="account-1")
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Ghi đè data_dir mặc định; có thể truyền nhiều lần.",
    )
    args = parser.parse_args()
    if args.matrix:
        if not args.backbones or not args.seeds or not args.datasets:
            parser.error("--matrix yêu cầu --backbones, --seeds và --datasets")
        overrides = {}
        for item in args.dataset_dir:
            if "=" not in item:
                parser.error(f"--dataset-dir phải có dạng DATASET=PATH: {item!r}")
            dataset, path = item.split("=", 1)
            overrides[dataset] = path
        generate_matrix(
            Path(args.template),
            Path(args.output_dir),
            args.git_commit,
            backbones=args.backbones,
            seeds=args.seeds,
            datasets=args.datasets,
            dataset_dirs=overrides,
            account_id=args.account_id,
        )
    else:
        generate(
            Path(args.registry),
            Path(args.template),
            Path(args.output_dir),
            args.git_commit,
        )
