"""Run independent Stage 1-4 priors concurrently, one seed per GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import shutil
from pathlib import Path
from queue import Queue


def _gpu_count() -> int:
    output = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    return len([line for line in output.splitlines() if line.strip()])


def _worker(
    gpu: int,
    queue: Queue,
    project: Path,
    output_root: Path,
    working_dir: Path,
    kaggle_input_root: Path,
    failures: list[dict],
) -> None:
    while True:
        run = queue.get()
        if run is None:
            queue.task_done()
            return
        run_id = run.get("run_id") or (
            f"{run['dataset_id']}__{run['backbone']}__db__seed_{run['seed']}"
        )
        output_dir = (
            output_root / run["backbone"] / run["dataset_id"]
            / f"seed_{run['seed']}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "stage1_4_params.yaml"
        if not config_path.exists():
            shutil.copy2(project / "params.yaml", config_path)
        log_path = output_dir / "stage1_4.log"
        command = [
            sys.executable, "-m", "pipelines.run_core_seed_experiment",
            "--config", str(config_path),
            "--dataset-id", run["dataset_id"],
            "--account-id", run.get("account_id", "dual-gpu-prior"),
            "--run-id", run_id,
            "--seed", str(run["seed"]),
            "--backbone", run["backbone"],
            "--data-dir", run["data_dir"],
            "--output-dir", str(output_dir),
            "--project-dir", str(project),
            "--working-dir", str(working_dir),
            "--kaggle-input-root", str(kaggle_input_root),
            "--stop-after-stage4",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        env["DISABLE_DVCLIVE"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["DVCLIVE_LOGLEVEL"] = "ERROR"
        env["DVC_NO_ANALYTICS"] = "1"
        print(
            f"[GPU {gpu}] Stage 1-4: {run_id}", flush=True
        )
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=project,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            failures.append(
                {"gpu": gpu, "run_id": run_id, "returncode": result.returncode}
            )
        queue.task_done()


def run(args: argparse.Namespace) -> None:
    if _gpu_count() < 2:
        raise RuntimeError("Cần accelerator GPU T4 x2.")
    runs = json.loads(Path(args.runs_file).read_text(encoding="utf-8"))
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs-file phải chứa JSON list không rỗng.")
    queue: Queue = Queue()
    for item in runs:
        queue.put(item)
    for _ in range(2):
        queue.put(None)
    failures: list[dict] = []
    threads = [
        threading.Thread(
            target=_worker,
            args=(
                gpu,
                queue,
                Path(args.project_dir).resolve(),
                Path(args.output_root).resolve(),
                Path(args.working_dir).resolve(),
                Path(args.kaggle_input_root).resolve(),
                failures,
            ),
        )
        for gpu in range(2)
    ]
    for thread in threads:
        thread.start()
    queue.join()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"Stage 1-4 worker failures: {failures}")
    print(f"Completed {len(runs)} Stage 1-4 priors on two GPUs.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", required=True)
    parser.add_argument("--project-dir", default=os.getcwd())
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--working-dir", default="/kaggle/working")
    parser.add_argument("--kaggle-input-root", default="/kaggle/input")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
