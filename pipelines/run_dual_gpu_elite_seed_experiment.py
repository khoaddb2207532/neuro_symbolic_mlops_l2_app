"""Run TB and DB GFlowNet -> Bayesian Elite concurrently on two Kaggle GPUs.

This runner assumes baseline/features/raw rules and heuristic Stage 5 outputs
already exist in a managed Kaggle input for the same dataset/backbone/seed.
Each loss branch gets an isolated output/config and a dedicated visible GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml

from pipelines.run_core_seed_experiment import (
    HEURISTICS,
    SUPPORTED_BACKBONES,
    _find_restorable_output,
    _safe_extract_tar,
)


LOSSES = ("tb", "db")


def _write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Không có kết quả để ghi CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _required_prior_paths(prior: Path, backbone: str) -> list[Path]:
    paths = [
        prior / "baseline_comparison" / backbone / "baseline_best.pth",
        prior / "02_features" / "val_features.pt",
        prior / "02_features" / "val_labels.pt",
        prior / "03_rules" / "raw_rules.pkl",
    ]
    paths.extend(
        prior / f"05_rules_model_{method}" / "rule_regularized_best.pth"
        for method in HEURISTICS
    )
    return paths


def _resolve_prior(args: argparse.Namespace, run_root: Path) -> Path:
    explicit = Path(args.prior_output_dir).resolve() if args.prior_output_dir else None
    if explicit is not None:
        prior = explicit
    else:
        found = _find_restorable_output(
            Path(args.kaggle_input_root),
            args.seed,
            args.backbone,
            args.dataset_id,
            args.prior_run_id,
        )
        if found is None:
            raise FileNotFoundError(
                "Không tìm thấy output stage trước trong Kaggle Input cho "
                f"run_id={args.prior_run_id!r}."
            )
        kind, source = found
        if kind == "directory":
            prior = source.resolve()
        else:
            prior = run_root / "restored_prior"
            if not prior.exists():
                _safe_extract_tar(source, prior)

    missing = [path for path in _required_prior_paths(prior, args.backbone) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Output stage trước thiếu artefact bắt buộc:\n- "
            + "\n- ".join(map(str, missing))
        )
    return prior


def _restore_partial_dual_run(args: argparse.Namespace, run_root: Path) -> None:
    if run_root.exists():
        return
    input_root = Path(args.kaggle_input_root)
    for manifest_path in sorted(input_root.rglob("dual_elite_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            int(manifest.get("seed", -1)) == args.seed
            and manifest.get("backbone") == args.backbone
            and manifest.get("dataset_id") == args.dataset_id
        ):
            print("Khôi phục dual-GPU run để resume từ:", manifest_path.parent)
            shutil.copytree(manifest_path.parent, run_root, dirs_exist_ok=True)
            return


def _link_directory(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        return
    try:
        destination.symlink_to(source.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(source, destination, dirs_exist_ok=True)


def _prepare_branch(
    *,
    project: Path,
    prior: Path,
    branch_dir: Path,
    loss_type: str,
    args: argparse.Namespace,
) -> Path:
    branch_dir.mkdir(parents=True, exist_ok=True)
    for name in ("baseline_comparison", "02_features", "03_rules"):
        _link_directory(prior / name, branch_dir / name)

    config = yaml.safe_load((project / args.config).read_text(encoding="utf-8"))
    config["seed"] = args.seed
    config["data_dir"] = args.data_dir
    config["output_dir"] = str(branch_dir)
    config["batch_size"] = args.batch_size
    config["num_workers"] = args.num_workers_per_gpu
    config["num_epochs"] = args.num_epochs
    config["patience"] = args.patience
    config["baseline_comparison"]["selected_architecture"] = args.backbone
    config["baseline_comparison"]["architectures"] = [args.backbone]
    config["gflownet"]["loss_type"] = loss_type
    config["gflownet"]["num_iterations"] = args.gflownet_iterations
    config.setdefault("rule_penalty_bayesian_elite", {})["K"] = args.mc_samples
    config["rule_penalty_bayesian_elite"]["sampler_checkpoint"] = (
        "gflownet_best_elite.pth"
    )
    config["rule_penalty_bayesian_elite"]["resume"] = True
    config_path = branch_dir / f"params_{loss_type}.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    metadata = {
        "seed": args.seed,
        "dataset_id": args.dataset_id,
        "backbone": args.backbone,
        "loss_type": loss_type,
        "gpu_index": 0 if loss_type == "tb" else 1,
        "prior_run_id": args.prior_run_id,
        "mc_samples": args.mc_samples,
    }
    metadata_path = branch_dir / "branch_metadata.json"
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved != metadata:
            raise ValueError(
                f"Resume branch {loss_type} sai identity: saved={saved}, current={metadata}"
            )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return config_path


def _worker(project: Path, config_path: Path, branch_dir: Path) -> None:
    filtered = branch_dir / "04_filtered_rules"
    elite = filtered / "gflownet_best_elite.pth"
    rule_order = filtered / "gflownet_rule_order.pkl"
    if elite.exists() and rule_order.exists():
        print("SKIP/RESUME Stage 4: đã có elite checkpoint + rule order.", flush=True)
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pipelines.stage4_select_rules_gflownet",
                "--config",
                str(config_path),
            ],
            cwd=project,
            check=True,
        )
    if not elite.exists() or not rule_order.exists():
        raise FileNotFoundError(
            f"Stage 4 không tạo đủ elite artefact trong {filtered}."
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.stage5_train_rule_bayesian_elite",
            "--config",
            str(config_path),
        ],
        cwd=project,
        check=True,
    )


def _stream(pipe, prefix: str, log_file) -> None:
    assert pipe is not None
    for line in iter(pipe.readline, ""):
        log_file.write(line)
        log_file.flush()
        print(f"[{prefix}] {line}", end="", flush=True)
    pipe.close()


def _launch_workers(
    project: Path,
    branch_configs: Dict[str, Path],
    branch_dirs: Dict[str, Path],
    run_root: Path,
) -> None:
    try:
        gpu_count = len(
            subprocess.check_output(["nvidia-smi", "-L"], text=True)
            .strip()
            .splitlines()
        )
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        gpu_count = 0
    if gpu_count < 2:
        raise RuntimeError(
            f"Notebook cần Kaggle accelerator 'GPU T4 x2'; chỉ thấy {gpu_count} GPU."
        )

    processes = {}
    streams = []
    logs = []
    for gpu_index, loss_type in enumerate(LOSSES):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable,
            "-m",
            "pipelines.run_dual_gpu_elite_seed_experiment",
            "--worker",
            "--project-dir",
            str(project),
            "--worker-config",
            str(branch_configs[loss_type]),
            "--worker-output-dir",
            str(branch_dirs[loss_type]),
        ]
        log_file = (run_root / f"worker_{loss_type}.log").open(
            "a", encoding="utf-8"
        )
        logs.append(log_file)
        process = subprocess.Popen(
            command,
            cwd=project,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes[loss_type] = process
        thread = threading.Thread(
            target=_stream,
            args=(process.stdout, loss_type.upper(), log_file),
            daemon=True,
        )
        thread.start()
        streams.append(thread)

    failures = {}
    for loss_type, process in processes.items():
        return_code = process.wait()
        if return_code:
            failures[loss_type] = return_code
    for thread in streams:
        thread.join()
    for log_file in logs:
        log_file.close()
    if failures:
        raise RuntimeError(f"Worker thất bại: {failures}")


def _evaluate(
    args: argparse.Namespace,
    prior: Path,
    branch_dirs: Dict[str, Path],
    run_root: Path,
) -> tuple[Path, Path]:
    # Import only after both training processes release their GPU contexts.
    import torch

    from pipelines.evaluate_saved_checkpoints import (
        _class_names,
        _metrics,
        _predict,
    )
    from src.data.dataset import create_dataloaders
    from src.models.cnn import ImageClassificationBaseline
    from src.utils.checkpoint import load_model_weights
    from src.utils.seed import set_seed

    specs = [
        (
            "cnn_baseline",
            prior / "baseline_comparison" / args.backbone / "baseline_best.pth",
        ),
        *[
            (
                method,
                prior / f"05_rules_model_{method}" / "rule_regularized_best.pth",
            )
            for method in HEURISTICS
        ],
    ]
    fixed_gfn = prior / "05_rules_model" / "rule_regularized_best.pth"
    if fixed_gfn.exists():
        specs.append(("gflownet_fixed_prior", fixed_gfn))
    specs.extend(
        (
            f"gflownet_{loss_type}_bayesian_elite",
            branch_dirs[loss_type]
            / "05b_rules_model_bayesian_elite"
            / "rule_regularized_best.pth",
        )
        for loss_type in LOSSES
    )
    missing = [path for _, path in specs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Thiếu checkpoint để so sánh:\n- " + "\n- ".join(map(str, missing))
        )

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, _, _, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers_per_gpu,
        seed=args.seed,
    )
    class_names = _class_names(args.data_dir)
    details = []
    rows = []
    for method, checkpoint in specs:
        model = ImageClassificationBaseline(
            architecture=args.backbone,
            num_classes=len(class_names),
            pretrained=False,
        )
        load_model_weights(model, str(checkpoint), device, required=True)
        model = model.to(device)
        labels, predictions = _predict(model, test_loader, device)
        metrics = _metrics(labels, predictions, class_names)
        details.append(
            {
                "seed": args.seed,
                "backbone": args.backbone,
                "method": method,
                "checkpoint": str(checkpoint),
                **metrics,
            }
        )
        rows.append(
            {
                "seed": args.seed,
                "backbone": args.backbone,
                "method": method,
                "test_accuracy": metrics["accuracy"],
                "test_macro_precision": metrics["macro_precision"],
                "test_macro_recall": metrics["macro_recall"],
                "test_f1_macro": metrics["macro_f1"],
                "test_weighted_f1": metrics["weighted_f1"],
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline = next(row for row in rows if row["method"] == "cnn_baseline")
    for row in rows:
        row["accuracy_delta_vs_baseline"] = (
            row["test_accuracy"] - baseline["test_accuracy"]
        )
        row["f1_delta_vs_baseline"] = (
            row["test_f1_macro"] - baseline["test_f1_macro"]
        )
    csv_path = run_root / f"seed_{args.seed}_dual_elite_comparison.csv"
    json_path = run_root / f"seed_{args.seed}_dual_elite_details.json"
    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return csv_path, json_path


def run(args: argparse.Namespace) -> None:
    project = Path(args.project_dir).resolve()
    run_root = Path(args.output_dir).resolve()
    _restore_partial_dual_run(args, run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    prior = _resolve_prior(args, run_root)
    manifest_path = run_root / "dual_elite_manifest.json"
    manifest = {
        "status": "running",
        "seed": args.seed,
        "dataset_id": args.dataset_id,
        "backbone": args.backbone,
        "prior_run_id": args.prior_run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    branch_dirs = {loss: run_root / loss for loss in LOSSES}
    branch_configs = {
        loss: _prepare_branch(
            project=project,
            prior=prior,
            branch_dir=branch_dirs[loss],
            loss_type=loss,
            args=args,
        )
        for loss in LOSSES
    }
    _launch_workers(project, branch_configs, branch_dirs, run_root)
    csv_path, json_path = _evaluate(args, prior, branch_dirs, run_root)

    manifest.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "comparison_csv": str(csv_path),
            "details_json": str(json_path),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive = shutil.make_archive(
        str(run_root.parent / f"seed_{args.seed}_dual_elite_artifacts"),
        "gztar",
        root_dir=run_root,
    )
    print("\nHoàn tất dual-GPU Bayesian Elite:")
    print(" -", csv_path)
    print(" -", json_path)
    print(" -", manifest_path)
    print(" -", archive)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-dir", help=argparse.SUPPRESS)
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset-id")
    parser.add_argument("--prior-run-id")
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES)
    parser.add_argument("--data-dir")
    parser.add_argument("--prior-output-dir")
    parser.add_argument("--output-dir")
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
    if parsed.worker:
        if not parsed.worker_config or not parsed.worker_output_dir:
            raise ValueError("Worker thiếu --worker-config/--worker-output-dir.")
        _worker(
            Path(parsed.project_dir).resolve(),
            Path(parsed.worker_config).resolve(),
            Path(parsed.worker_output_dir).resolve(),
        )
    else:
        required = {
            "seed": parsed.seed,
            "dataset_id": parsed.dataset_id,
            "prior_run_id": parsed.prior_run_id,
            "backbone": parsed.backbone,
            "data_dir": parsed.data_dir,
            "output_dir": parsed.output_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Thiếu tham số: {missing}")
        run(parsed)
