"""Generate Stage 1-4 seed-pair notebooks and per-seed DB Stage-5 notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATA_DIRS = {
    "culture-a": "/kaggle/input/datasets/dangduykhoab2207532/"
    "vietnamese-cultural-dataset/vietnamese_cultural_dataset",
    "culture-b": "/kaggle/working",
}
REPO_URL = "https://github.com/khoaddb2207532/neuro_symbolic_mlops_l2_app.git"


def _code(source: str, tag: str | None = None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [tag]} if tag else {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _base_cells(git_commit: str) -> list[dict]:
    clone = f'''import os
import shutil
import subprocess
from pathlib import Path

GIT_COMMIT = {git_commit!r}
PROJECT = Path('/kaggle/working/neuro_symbolic_mlops_l2_app')
if not (PROJECT / '.git').exists():
    if PROJECT.exists():
        # Chỉ dọn đúng thư mục clone dở của notebook trước đó.
        shutil.rmtree(PROJECT)
    git_env = os.environ.copy()
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('GITHUB_TOKEN')
    except Exception as error:
        token = None
        print('Không đọc được Kaggle secret GITHUB_TOKEN:', type(error).__name__)
    if token:
        # Truyền auth qua environment để token không xuất hiện trong command/traceback.
        git_env['GIT_CONFIG_COUNT'] = '1'
        git_env['GIT_CONFIG_KEY_0'] = 'http.extraHeader'
        git_env['GIT_CONFIG_VALUE_0'] = f'Authorization: Bearer {{token}}'
        print('GitHub authentication: Kaggle secret GITHUB_TOKEN')
    else:
        print('GitHub authentication: none (chỉ hoạt động nếu repo public)')
    clone = subprocess.run(
        ['git', 'clone', '--filter=blob:none', {REPO_URL!r}, str(PROJECT)],
        env=git_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if clone.returncode:
        raise RuntimeError(
            'git clone thất bại. Hãy bật Kaggle Internet và cấu hình '
            'GITHUB_TOKEN nếu repo private. Git stderr: ' + clone.stderr.strip()
        )
subprocess.run(['git', 'fetch', 'origin', GIT_COMMIT, '--depth', '1'], cwd=PROJECT, check=True)
subprocess.run(['git', 'checkout', '--detach', GIT_COMMIT], cwd=PROJECT, check=True)
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip() == GIT_COMMIT
'''
    setup = '''%cd /kaggle/working/neuro_symbolic_mlops_l2_app
!pip install -q torchgfn tensordict dvclive dvc openpyxl
!nvidia-smi -L
import torch
assert torch.cuda.device_count() >= 2, 'Hãy chọn Kaggle Accelerator: GPU T4 x2'
'''
    warning_control = '''import logging
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['DISABLE_DVCLIVE'] = '1'
os.environ['DVCLIVE_LOGLEVEL'] = 'ERROR'
os.environ['DVC_NO_ANALYTICS'] = '1'
logging.getLogger('dvclive').setLevel(logging.ERROR)
logging.getLogger('dvc').setLevel(logging.ERROR)
print('Dual-GPU mode: DVCLive disabled; warnings suppressed.')
'''
    return [
        _code(clone, "checkout"),
        _code(setup, "dependencies"),
        _code(warning_control, "warning-control"),
    ]


def _dataset_setup(datasets: str | list[str]) -> dict:
    dataset_list = [datasets] if isinstance(datasets, str) else list(datasets)
    if "culture-b" not in dataset_list:
        return _code(
            "from pathlib import Path\n"
            + "\n".join(
                f"assert Path({DATA_DIRS[dataset]!r}).exists()"
                for dataset in dataset_list
            )
            + "\n",
            "dataset-setup",
        )
    return _code(
        '''from pathlib import Path
source = Path('/kaggle/input/datasets/utkarshsaxenadn/fast-food-classification-dataset/Fast Food Classification V2')
for destination, origin in {'train': 'Train', 'test': 'Test', 'val': 'Valid'}.items():
    path = Path('/kaggle/working') / destination
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        raise FileExistsError(f'Không ghi đè path thật: {path}')
    path.symlink_to(source / origin, target_is_directory=True)
''',
        "dataset-setup",
    )


def _notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def generate(
    output_dir: Path,
    git_commit: str,
    *,
    backbones: list[str],
    datasets: list[str],
    seeds: list[int],
    seeds_per_prior_notebook: int = 2,
) -> tuple[list[Path], list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_paths: list[Path] = []
    stage5_paths: list[Path] = []
    for backbone in backbones:
        all_prior_runs = [
            {
                "dataset_id": dataset,
                "data_dir": DATA_DIRS[dataset],
                "seed": seed,
                "backbone": backbone,
                "run_id": f"{dataset}__{backbone}__db__seed_{seed}",
            }
            for dataset in datasets
            for seed in seeds
        ]
        for start in range(0, len(all_prior_runs), seeds_per_prior_notebook):
            runs = all_prior_runs[start:start + seeds_per_prior_notebook]
            params = f"RUNS = {runs!r}\n"
            runner = '''import json, subprocess, sys
from pathlib import Path
runs_file = Path('/kaggle/working/stage1_4_runs.json')
runs_file.write_text(json.dumps(RUNS), encoding='utf-8')
subprocess.run([
    sys.executable, '-m', 'pipelines.run_dual_gpu_stage1_4_multi_seed',
    '--runs-file', str(runs_file),
    '--project-dir', str(PROJECT),
    '--output-root', '/kaggle/working/stage1_4_prior_runs',
    '--working-dir', '/kaggle/working',
    '--kaggle-input-root', '/kaggle/input',
], cwd=PROJECT, check=True)
'''
            labels = [f"{run['dataset_id']}-s{run['seed']}" for run in runs]
            notebook = _notebook([
                _markdown(
                    f"# Dual-GPU Stage 1-4 prior — {backbone} / {', '.join(labels)}\n\n"
                    "Mỗi GPU chạy trọn Stage 1→4 của một run độc lập."
                ),
                _code(params, "prior-parameters"),
                _dataset_setup(sorted({run["dataset_id"] for run in runs})),
                *_base_cells(git_commit),
                _code(runner, "dual-gpu-stage1-4"),
            ])
            name = f"prior_stage1_4_{backbone}_{'_'.join(labels)}.ipynb"
            path = output_dir / "stage1_4_prior" / backbone / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
            prior_paths.append(path)

        for dataset in datasets:
            data_dir = DATA_DIRS[dataset]
            for seed in seeds:
                run_id = f"{dataset}__{backbone}__db__seed_{seed}"
                params = (
                    f"BACKBONE = {backbone!r}\nDATASET_ID = {dataset!r}\n"
                    f"DATA_DIR = {data_dir!r}\nSEED = {seed}\nPRIOR_RUN_ID = {run_id!r}\n"
                    "MC_SAMPLES = 32\n"
                )
                runner = '''import subprocess, sys
from pathlib import Path
output = Path('/kaggle/working/db_stage5_runs') / BACKBONE / DATASET_ID / f'seed_{SEED}'
subprocess.run([
    sys.executable, '-m', 'pipelines.run_dual_gpu_db_stage5_seed',
    '--config', 'params.yaml',
    '--seed', str(SEED),
    '--dataset-id', DATASET_ID,
    '--prior-run-id', PRIOR_RUN_ID,
    '--backbone', BACKBONE,
    '--data-dir', DATA_DIR,
    '--output-dir', str(output),
    '--project-dir', str(PROJECT),
    '--working-dir', '/kaggle/working',
    '--kaggle-input-root', '/kaggle/input',
    '--mc-samples', str(MC_SAMPLES),
], cwd=PROJECT, check=True)
'''
                notebook = _notebook([
                    _markdown(
                        f"# Dual-GPU DB Stage 5 — {backbone} / {dataset} / seed {seed}\n\n"
                        "Add Output của notebook Stage 1-4 tương ứng làm Kaggle Input. "
                        "Hai GPU chia sẻ hàng đợi gồm DB fixed, DB Bayesian, DB Bayesian Elite, Random, Top-K và Greedy."
                    ),
                    _code(params, "stage5-parameters"),
                    _dataset_setup(dataset),
                    *_base_cells(git_commit),
                    _code(runner, "dual-gpu-db-stage5"),
                ])
                name = f"db_stage5_{backbone}_{dataset}_seed_{seed}.ipynb"
                path = output_dir / "stage5_db" / backbone / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
                stage5_paths.append(path)
    print(f"Generated {len(prior_paths)} Stage 1-4 notebooks and {len(stage5_paths)} Stage-5 notebooks.")
    return prior_paths, stage5_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="generated_two_phase_dual_gpu_notebooks")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--backbones", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATA_DIRS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--seeds-per-prior-notebook", type=int, default=2)
    args = parser.parse_args()
    generate(
        Path(args.output_dir), args.git_commit,
        backbones=args.backbones,
        datasets=args.datasets,
        seeds=args.seeds,
        seeds_per_prior_notebook=args.seeds_per_prior_notebook,
    )
