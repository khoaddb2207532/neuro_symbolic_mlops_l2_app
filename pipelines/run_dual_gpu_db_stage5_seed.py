"""Train six DB Stage-5 methods through a shared two-GPU work queue."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue

import yaml

from pipelines.run_core_seed_experiment import (
    _find_restorable_output,
    _safe_extract_tar,
    _write_config,
)


METHODS = (
    "gflownet_db_bayesian",
    "gflownet_db_bayesian_elite",
    "gflownet_db_fixed",
    "random",
    "topk_confidence",
    "greedy_coverage",
)
STAGE5_DIRS = (
    "05_rules_model",
    "05b_rules_model_bayesian",
    "05b_rules_model_bayesian_elite",
    "04_filtered_rules_random",
    "05_rules_model_random",
    "04_filtered_rules_topk_confidence",
    "05_rules_model_topk_confidence",
    "04_filtered_rules_greedy_coverage",
    "05_rules_model_greedy_coverage",
)


def _restore_stage5_if_available(
    input_root: Path,
    output: Path,
    *,
    seed: int,
    backbone: str,
    dataset_id: str,
) -> None:
    for manifest_path in sorted(input_root.rglob("stage5_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("seed") != seed
            or manifest.get("backbone") != backbone
            or manifest.get("dataset_id") != dataset_id
        ):
            continue
        source_root = manifest_path.parent
        for name in STAGE5_DIRS:
            source = source_root / name
            destination = output / name
            if source.is_dir() and not destination.exists():
                shutil.copytree(source, destination)
        print("Restored partial Stage 5 output from", source_root, flush=True)
        return


def _link_prior(prior: Path, output: Path) -> None:
    for name in ("baseline_comparison", "02_features", "03_rules", "04_filtered_rules"):
        source = prior / name
        destination = output / name
        if not source.exists():
            raise FileNotFoundError(f"Prior thiếu {source}")
        if destination.exists() or destination.is_symlink():
            continue
        destination.symlink_to(source, target_is_directory=True)


def _method_command(method: str, config: Path, seed: int) -> list[str]:
    prefix = [sys.executable, "-m"]
    if method == "gflownet_db_fixed":
        return prefix + ["pipelines.stage5_train_rule_regularized", "--config", str(config)]
    if method == "gflownet_db_bayesian":
        return prefix + ["pipelines.stage5_train_rule_bayesian", "--config", str(config)]
    if method == "gflownet_db_bayesian_elite":
        return prefix + ["pipelines.stage5_train_rule_bayesian_elite", "--config", str(config)]
    return prefix + [
        "pipelines.stage5_train_rule_regularized_heuristics",
        "--config", str(config), "--methods", method,
        "--random_seed", str(seed),
    ]


def _checkpoint(output: Path, method: str) -> Path:
    names = {
        "gflownet_db_fixed": "05_rules_model",
        "gflownet_db_bayesian": "05b_rules_model_bayesian",
        "gflownet_db_bayesian_elite": "05b_rules_model_bayesian_elite",
        "random": "05_rules_model_random",
        "topk_confidence": "05_rules_model_topk_confidence",
        "greedy_coverage": "05_rules_model_greedy_coverage",
    }
    return output / names[method] / "rule_regularized_best.pth"


def _gpu_worker(
    gpu: int,
    queue: Queue,
    project: Path,
    config: Path,
    output: Path,
    seed: int,
    failures: list[dict],
) -> None:
    while True:
        method = queue.get()
        if method is None:
            queue.task_done()
            return
        checkpoint = _checkpoint(output, method)
        if checkpoint.exists():
            print(f"[GPU {gpu}] SKIP {method}: {checkpoint}", flush=True)
            queue.task_done()
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        env["DISABLE_DVCLIVE"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["DVCLIVE_LOGLEVEL"] = "ERROR"
        env["DVC_NO_ANALYTICS"] = "1"
        log_path = output / f"stage5_{method}.log"
        print(f"[GPU {gpu}] START {method}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                _method_command(method, config, seed),
                cwd=project,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode or not checkpoint.exists():
            failures.append(
                {"gpu": gpu, "method": method, "returncode": result.returncode}
            )
        queue.task_done()


def run(args: argparse.Namespace) -> None:
    gpu_count = len(
        subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
    )
    if gpu_count < 2:
        raise RuntimeError("Cần accelerator GPU T4 x2.")
    project = Path(args.project_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _restore_stage5_if_available(
        Path(args.kaggle_input_root),
        output,
        seed=args.seed,
        backbone=args.backbone,
        dataset_id=args.dataset_id,
    )
    located = _find_restorable_output(
        Path(args.kaggle_input_root),
        args.seed,
        args.backbone,
        args.dataset_id,
        args.prior_run_id,
    )
    if located is None:
        raise FileNotFoundError(f"Không tìm thấy Stage 1-4 prior {args.prior_run_id!r}")
    kind, source = located
    if kind == "archive":
        prior = output / "restored_prior"
        if not prior.exists():
            _safe_extract_tar(source, prior)
    else:
        prior = source.resolve()
    _link_prior(prior, output)
    config_path = output / "stage5_params.yaml"
    shutil.copy2(project / args.config, config_path)
    config = _write_config(
        config_path,
        seed=args.seed,
        backbone=args.backbone,
        data_dir=Path(args.data_dir),
        output_dir=output,
    )
    config.setdefault("rule_penalty_bayesian", {})["K"] = args.mc_samples
    config.setdefault("rule_penalty_bayesian_elite", {})["K"] = args.mc_samples
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    required = [
        output / "04_filtered_rules" / name
        for name in (
            "selected_rules.pkl",
            "gflownet_rule_order.pkl",
            "gflownet_best_diverse.pth",
            "gflownet_best_elite.pth",
        )
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Prior thiếu DB sampler:\n- " + "\n- ".join(map(str, missing)))

    manifest = {
        "status": "running",
        "seed": args.seed,
        "dataset_id": args.dataset_id,
        "backbone": args.backbone,
        "prior_run_id": args.prior_run_id,
        "methods": list(METHODS),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output / "stage5_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    queue: Queue = Queue()
    for method in METHODS:
        queue.put(method)
    queue.put(None)
    queue.put(None)
    failures: list[dict] = []
    threads = [
        threading.Thread(
            target=_gpu_worker,
            args=(gpu, queue, project, config_path, output, args.seed, failures),
        )
        for gpu in range(2)
    ]
    for thread in threads:
        thread.start()
    queue.join()
    for thread in threads:
        thread.join()
    if failures:
        manifest.update({"status": "failed", "failures": failures})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise RuntimeError(f"Stage 5 failures: {failures}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["DISABLE_DVCLIVE"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    env["DVCLIVE_LOGLEVEL"] = "ERROR"
    env["DVC_NO_ANALYTICS"] = "1"
    subprocess.run(
        [
            sys.executable, "-m", "pipelines.evaluate_saved_checkpoints",
            "--config", str(config_path), "--include-bayesian",
            "--include-bayesian-elite",
        ],
        cwd=project,
        env=env,
        check=True,
    )
    manifest.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": str(output / "exact_test_metrics_summary.csv"),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = shutil.make_archive(
        str(Path(args.working_dir) / f"seed_{args.seed}_db_stage5"),
        "gztar",
        root_dir=output,
    )
    print("Stage 5 DB complete:", output / "exact_test_metrics_summary.csv")
    print("Archive:", archive)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--prior-run-id", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-dir", default=os.getcwd())
    parser.add_argument("--working-dir", default="/kaggle/working")
    parser.add_argument("--kaggle-input-root", default="/kaggle/input")
    parser.add_argument("--mc-samples", type=int, default=32)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
