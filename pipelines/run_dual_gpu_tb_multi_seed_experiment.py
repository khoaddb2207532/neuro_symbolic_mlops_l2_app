"""Run several TB GFlowNet seeds concurrently on exactly two GPUs.

Each GPU owns one persistent queue of seeds.  A seed runs Stage 4 TB followed by
the fixed-rule and Bayesian-TB Stage 5 variants.  Stage 1--3 artifacts are read
from a prior managed run and never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import yaml

from pipelines.run_core_seed_experiment import (
    SUPPORTED_BACKBONES,
    _find_restorable_output,
    _safe_extract_tar,
)


METHODS = {
    "cnn_baseline": ("baseline", "baseline_best.pth"),
    "gflownet_tb_fixed": ("05_rules_model", "rule_regularized_best.pth"),
    "gflownet_tb_bayesian": (
        "05b_rules_model_bayesian",
        "rule_regularized_best.pth",
    ),
}
STAGE4_FILES = (
    "selected_rules.pkl",
    "gflownet_rule_order.pkl",
    "gflownet_best_diverse.pth",
)


def _resume_enabled(run: dict[str, Any]) -> bool:
    """Whether this run may restore and resume artifacts from an older session."""
    return bool(run.get("resume", True))


def partition_runs(runs: list[dict[str, Any]], gpu_count: int = 2) -> list[list[dict[str, Any]]]:
    """Deterministically assign runs round-robin to fixed GPU queues."""
    if gpu_count != 2:
        raise ValueError("Workflow này yêu cầu đúng 2 hàng đợi GPU.")
    if not runs:
        raise ValueError("Danh sách run/seed rỗng.")
    queues = [[] for _ in range(gpu_count)]
    for index, run in enumerate(runs):
        queues[index % gpu_count].append(run)
    return queues


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Không có dữ liệu để ghi {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _link_or_copy(source: Path, destination: Path) -> None:
    # Saved Kaggle outputs may contain a symlink pointing at an older Input
    # mount. Recreate it against the current prior instead of trusting it.
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        return
    try:
        destination.symlink_to(source.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(source, destination)


def _resolve_prior(args: argparse.Namespace, run: dict[str, Any], run_dir: Path) -> Path:
    found = _find_restorable_output(
        Path(args.kaggle_input_root),
        int(run["seed"]),
        args.backbone,
        run["dataset_id"],
        run["prior_run_id"],
    )
    if found is None:
        raise FileNotFoundError(
            f"Không tìm thấy prior Stage 1-3 cho {run['prior_run_id']!r}."
        )
    kind, source = found
    if kind == "directory":
        prior = source.resolve()
    else:
        prior = run_dir / "restored_prior"
        if not prior.exists():
            _safe_extract_tar(source, prior)
    required = [
        prior / "baseline_comparison" / args.backbone / "baseline_best.pth",
        prior / "02_features" / "val_features.pt",
        prior / "02_features" / "val_labels.pt",
        prior / "03_rules" / "raw_rules.pkl",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Prior thiếu artefact:\n- " + "\n- ".join(map(str, missing)))
    return prior


def _restore_run(args: argparse.Namespace, run: dict[str, Any], run_dir: Path) -> None:
    """Restore a matching interrupted TB run from a read-only Kaggle Input."""
    if run_dir.exists():
        return
    input_root = Path(args.kaggle_input_root)
    for manifest_path in sorted(input_root.rglob("tb_run_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        expected = {
            "dataset_id": run["dataset_id"],
            "seed": int(run["seed"]),
            "backbone": args.backbone,
            "loss_type": "tb",
            "prior_run_id": run["prior_run_id"],
        }
        if all(manifest.get(key) == value for key, value in expected.items()):
            print(f"Restore seed {run['seed']} từ {manifest_path.parent}", flush=True)
            shutil.copytree(manifest_path.parent, run_dir, dirs_exist_ok=True)
            return


def _prepare_config(
    args: argparse.Namespace,
    run: dict[str, Any],
    prior: Path,
    run_dir: Path,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("baseline_comparison", "02_features", "03_rules"):
        _link_or_copy(prior / name, run_dir / name)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(args.project_dir) / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.update(
        seed=int(run["seed"]),
        data_dir=run["data_dir"],
        output_dir=str(run_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers_per_gpu,
        num_epochs=args.num_epochs,
        patience=args.patience,
    )
    config["baseline_comparison"]["selected_architecture"] = args.backbone
    config["baseline_comparison"]["architectures"] = [args.backbone]
    config["gflownet"]["loss_type"] = "tb"
    config["gflownet"]["num_iterations"] = args.gflownet_iterations
    resume = _resume_enabled(run)
    config.setdefault("rule_penalty", {})["resume"] = resume
    bayesian = config.setdefault("rule_penalty_bayesian", {})
    bayesian.update(
        K=args.mc_samples,
        sampler_checkpoint="gflownet_best_diverse.pth",
        resume=resume,
    )
    destination = run_dir / "params_tb.yaml"
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return destination


def _run_module(project: Path, module: str, config: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", module, "--config", str(config)],
        cwd=project,
        check=True,
    )


def _train_seed(args: argparse.Namespace, run: dict[str, Any], gpu_index: int) -> Path:
    project = Path(args.project_dir).resolve()
    run_dir = (
        Path(args.output_dir).resolve()
        / run["dataset_id"]
        / args.backbone
        / f"seed_{int(run['seed'])}"
    )
    if _resume_enabled(run):
        _restore_run(args, run, run_dir)
    else:
        print(
            f"Seed {run['seed']} chạy mới: bỏ qua restore từ Kaggle Input.",
            flush=True,
        )
    prior = _resolve_prior(args, run, run_dir)
    config = _prepare_config(args, run, prior, run_dir)
    manifest_path = run_dir / "tb_run_manifest.json"
    identity = {
        "dataset_id": run["dataset_id"],
        "seed": int(run["seed"]),
        "backbone": args.backbone,
        "loss_type": "tb",
        "gpu_index": gpu_index,
        "prior_run_id": run["prior_run_id"],
    }
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in identity.items():
            if saved.get(key) != value:
                raise ValueError(f"Resume sai identity tại {manifest_path}: {key}")
    manifest = {
        **identity,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    filtered = run_dir / "04_filtered_rules"
    if not all((filtered / name).exists() for name in STAGE4_FILES):
        _run_module(project, "pipelines.stage4_select_rules_gflownet", config)
    if not all((filtered / name).exists() for name in STAGE4_FILES):
        raise FileNotFoundError(f"Stage 4 TB chưa tạo đủ artefact tại {filtered}.")

    fixed = run_dir / "05_rules_model" / "rule_regularized_best.pth"
    if not fixed.exists():
        _run_module(project, "pipelines.stage5_train_rule_regularized", config)
    bayesian = run_dir / "05b_rules_model_bayesian" / "rule_regularized_best.pth"
    if not bayesian.exists():
        _run_module(project, "pipelines.stage5_train_rule_bayesian_tb", config)
    if not fixed.exists() or not bayesian.exists():
        raise FileNotFoundError(f"Stage 5 TB chưa tạo đủ checkpoint tại {run_dir}.")

    comparison = _evaluate_seed(args, run, prior, run_dir)
    manifest.update(
        status="complete",
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        comparison_csv=str(comparison),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return comparison


def _evaluate_seed(
    args: argparse.Namespace,
    run: dict[str, Any],
    prior: Path,
    run_dir: Path,
) -> Path:
    import torch

    from pipelines.evaluate_saved_checkpoints import _class_names, _metrics, _predict
    from src.data.dataset import create_dataloaders
    from src.models.cnn import ImageClassificationBaseline
    from src.utils.checkpoint import load_model_weights
    from src.utils.seed import set_seed

    checkpoints = {
        "cnn_baseline": prior
        / "baseline_comparison"
        / args.backbone
        / "baseline_best.pth",
        "gflownet_tb_fixed": run_dir
        / "05_rules_model"
        / "rule_regularized_best.pth",
        "gflownet_tb_bayesian": run_dir
        / "05b_rules_model_bayesian"
        / "rule_regularized_best.pth",
    }
    set_seed(int(run["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, test_loader = create_dataloaders(
        run["data_dir"],
        batch_size=args.batch_size,
        num_workers=args.num_workers_per_gpu,
        seed=int(run["seed"]),
    )
    class_names = _class_names(run["data_dir"])
    rows = []
    for method, checkpoint in checkpoints.items():
        model = ImageClassificationBaseline(
            architecture=args.backbone,
            num_classes=len(class_names),
            pretrained=False,
        )
        load_model_weights(model, str(checkpoint), device, required=True)
        model.to(device)
        labels, predictions = _predict(model, test_loader, device)
        metrics = _metrics(labels, predictions, class_names)
        rows.append(
            {
                "dataset_id": run["dataset_id"],
                "backbone": args.backbone,
                "seed": int(run["seed"]),
                "method": method,
                "test_accuracy": metrics["accuracy"],
                "test_f1_macro": metrics["macro_f1"],
                "test_macro_precision": metrics["macro_precision"],
                "test_macro_recall": metrics["macro_recall"],
            }
        )
        del model
        torch.cuda.empty_cache()
    baseline = next(row for row in rows if row["method"] == "cnn_baseline")
    for row in rows:
        row["accuracy_delta_vs_baseline"] = row["test_accuracy"] - baseline["test_accuracy"]
        row["f1_delta_vs_baseline"] = row["test_f1_macro"] - baseline["test_f1_macro"]
    path = run_dir / f"seed_{int(run['seed'])}_tb_comparison.csv"
    _write_csv(path, rows)
    return path


def _aggregate(output_dir: Path, comparison_paths: list[Path]) -> tuple[Path, Path]:
    detail = []
    for path in comparison_paths:
        with path.open(newline="", encoding="utf-8") as file:
            detail.extend(csv.DictReader(file))
    numeric = (
        "test_accuracy",
        "test_f1_macro",
        "test_macro_precision",
        "test_macro_recall",
        "accuracy_delta_vs_baseline",
        "f1_delta_vs_baseline",
    )
    for row in detail:
        row["seed"] = int(row["seed"])
        for key in numeric:
            row[key] = float(row[key])
    detail.sort(key=lambda row: (row["dataset_id"], row["backbone"], row["seed"], row["method"]))
    detail_path = output_dir / "tb_all_seed_comparison.csv"
    _write_csv(detail_path, detail)
    summary = []
    groups = sorted({(row["dataset_id"], row["backbone"], row["method"]) for row in detail})
    for dataset_id, backbone, method in groups:
        rows = [
            row for row in detail
            if (row["dataset_id"], row["backbone"], row["method"])
            == (dataset_id, backbone, method)
        ]
        item: dict[str, Any] = {
            "dataset_id": dataset_id,
            "backbone": backbone,
            "method": method,
            "n_seeds": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda x: x["seed"])),
        }
        for key in numeric:
            values = [row[key] for row in rows]
            item[f"{key}_mean"] = mean(values)
            item[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    summary_path = output_dir / "tb_all_seed_summary.csv"
    _write_csv(summary_path, summary)
    markdown = [
        "# So sánh GFlowNet-TB qua nhiều seed",
        "",
        "| Dataset | Model | Method | Seeds | Accuracy (mean±std) | Macro-F1 (mean±std) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary:
        markdown.append(
            f"| {row['dataset_id']} | {row['backbone']} | {row['method']} | "
            f"{row['n_seeds']} | {row['test_accuracy_mean']:.4f}±{row['test_accuracy_std']:.4f} | "
            f"{row['test_f1_macro_mean']:.4f}±{row['test_f1_macro_std']:.4f} |"
        )
    (output_dir / "tb_all_seed_summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return detail_path, summary_path


def _gpu_queue(args: argparse.Namespace, gpu_index: int, runs: list[dict[str, Any]], results: list[Path], errors: list[BaseException]) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["PYTHONUNBUFFERED"] = "1"
    for run in runs:
        command = [
            sys.executable,
            "-m",
            "pipelines.run_dual_gpu_tb_multi_seed_experiment",
            "--worker-run-json",
            json.dumps(run),
            "--gpu-index",
            str(gpu_index),
            *args.forwarded_args,
        ]
        try:
            subprocess.run(command, cwd=args.project_dir, env=env, check=True)
            results.append(
                Path(args.output_dir).resolve()
                / run["dataset_id"]
                / args.backbone
                / f"seed_{int(run['seed'])}"
                / f"seed_{int(run['seed'])}_tb_comparison.csv"
            )
        except BaseException as error:  # preserve the first worker failure
            errors.append(error)
            return


def run_controller(args: argparse.Namespace) -> None:
    runs = json.loads(Path(args.runs_json).read_text(encoding="utf-8"))
    required = {"dataset_id", "data_dir", "seed", "prior_run_id"}
    for run in runs:
        if not required.issubset(run):
            raise ValueError(f"Run thiếu field {sorted(required - set(run))}: {run}")
    try:
        gpu_count = len(subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines())
    except (OSError, subprocess.CalledProcessError):
        gpu_count = 0
    if gpu_count < 2:
        raise RuntimeError(f"Cần 2 GPU, chỉ phát hiện {gpu_count}.")
    queues = partition_runs(runs)
    args.forwarded_args = _forwarded_args(args)
    results: list[Path] = []
    errors: list[BaseException] = []
    threads = [
        threading.Thread(target=_gpu_queue, args=(args, index, queue, results, errors))
        for index, queue in enumerate(queues)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError(f"GPU worker thất bại: {errors[0]}") from errors[0]
    if len(results) != len(runs):
        raise RuntimeError(f"Thiếu kết quả seed: {len(results)}/{len(runs)}")
    detail, summary = _aggregate(Path(args.output_dir).resolve(), results)
    print(f"Bảng chi tiết: {detail}\nBảng tổng hợp: {summary}")


def _forwarded_args(args: argparse.Namespace) -> list[str]:
    pairs = {
        "--config": args.config,
        "--backbone": args.backbone,
        "--output-dir": args.output_dir,
        "--project-dir": args.project_dir,
        "--kaggle-input-root": args.kaggle_input_root,
        "--mc-samples": args.mc_samples,
        "--gflownet-iterations": args.gflownet_iterations,
        "--num-epochs": args.num_epochs,
        "--patience": args.patience,
        "--batch-size": args.batch_size,
        "--num-workers-per-gpu": args.num_workers_per_gpu,
    }
    result = []
    for flag, value in pairs.items():
        result.extend((flag, str(value)))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-json")
    parser.add_argument("--worker-run-json", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--backbone", required=True, choices=SUPPORTED_BACKBONES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-dir", default=os.getcwd())
    parser.add_argument("--kaggle-input-root", default="/kaggle/input")
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--gflownet-iterations", type=int, default=5000)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers-per-gpu", type=int, default=2)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.worker_run_json:
        _train_seed(parsed, json.loads(parsed.worker_run_json), parsed.gpu_index)
    elif parsed.runs_json:
        run_controller(parsed)
    else:
        raise SystemExit("Cần --runs-json ở controller.")
