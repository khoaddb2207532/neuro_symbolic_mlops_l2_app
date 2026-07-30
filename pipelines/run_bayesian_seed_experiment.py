"""Khôi phục output GFlowNet theo seed và chạy riêng Stage 5 Bayesian.

Module dành cho Kaggle Add Input. Nó không chạy lại baseline, feature
extraction, rule extraction hoặc GFlowNet; mọi artefact bắt buộc phải có trong
output notebook seed đã Save Version.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from pipelines.run_core_seed_experiment import (
    SUPPORTED_BACKBONES,
    _first_report,
    _report_metrics,
    _required_dataset_splits,
    _restore_if_available,
    _write_config,
)


def _require_files(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Kaggle Input thiếu artefact bắt buộc cho Bayesian Stage 5:\n- "
            + "\n- ".join(map(str, missing))
        )


def _sampler_checkpoint(filtered_dir: Path, preferred_name: str) -> Path:
    preferred = filtered_dir / preferred_name
    if preferred.exists():
        return preferred
    fallback = filtered_dir / "gflownet_best_converged.pth"
    if fallback.exists():
        print(f"Không có {preferred.name}; dùng fallback {fallback.name}.")
        return fallback
    raise FileNotFoundError(
        f"Không tìm thấy {preferred.name} hoặc {fallback.name} trong "
        f"{filtered_dir}."
    )


def _run_stage5(project: Path, config_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "pipelines.stage5_train_rule_bayesian",
        "--config",
        str(config_path),
    ]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=project, check=True)


def run(args: argparse.Namespace) -> None:
    project = Path(args.project_dir).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project / config_path
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    input_root = Path(args.kaggle_input_root)
    working_dir = Path(args.working_dir)

    _required_dataset_splits(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chỉ restore candidate có tên outputs_seed_<seed>/archive seed_<seed>.
    # Hàm restore đồng thời xác minh metadata seed + backbone.
    _restore_if_available(
        input_root,
        output_dir,
        args.seed,
        args.backbone,
    )

    config = _write_config(
        config_path,
        seed=args.seed,
        backbone=args.backbone,
        data_dir=data_dir,
        output_dir=output_dir,
    )
    config.setdefault("rule_penalty_bayesian", {})["K"] = args.mc_samples
    config["rule_penalty_bayesian"]["sampler_checkpoint"] = args.sampler_checkpoint
    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    baseline_checkpoint = (
        output_dir
        / "baseline_comparison"
        / args.backbone
        / "baseline_best.pth"
    )
    filtered_dir = output_dir / "04_filtered_rules"
    rule_order = filtered_dir / "gflownet_rule_order.pkl"
    sampler = _sampler_checkpoint(filtered_dir, args.sampler_checkpoint)
    _require_files([baseline_checkpoint, rule_order, sampler])

    metadata = {
        "seed": args.seed,
        "backbone": args.backbone,
        "loss_type": "db",
        "bayesian_mc_samples": args.mc_samples,
        "sampler_checkpoint": sampler.name,
    }
    (output_dir / "bayesian_experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_dir = output_dir / "05b_rules_model_bayesian"
    report = _first_report(save_dir)
    if report is not None:
        print("SKIP Bayesian Stage 5: đã có report", report)
    else:
        print(
            f"Chạy Bayesian Stage 5 | seed={args.seed} | "
            f"backbone={args.backbone} | sampler={sampler.name} | "
            f"MC K={args.mc_samples}",
            flush=True,
        )
        _run_stage5(project, config_path)
        report = _first_report(save_dir)
        if report is None:
            raise FileNotFoundError(
                "Bayesian Stage 5 hoàn tất nhưng không tạo classification report."
            )

    accuracy, f1_macro = _report_metrics(report)
    results_path = working_dir / f"seed_{args.seed}_bayesian_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "backbone",
                "method",
                "test_accuracy",
                "test_f1_macro",
                "mc_samples",
                "sampler_checkpoint",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seed": args.seed,
                "backbone": args.backbone,
                "method": "gflownet_db_bayesian",
                "test_accuracy": accuracy,
                "test_f1_macro": f1_macro,
                "mc_samples": args.mc_samples,
                "sampler_checkpoint": sampler.name,
            }
        )

    archive = shutil.make_archive(
        str(working_dir / f"seed_{args.seed}_bayesian_artifacts"),
        "gztar",
        root_dir=save_dir,
    )
    print("\nBayesian Stage 5 hoàn tất:")
    print(" -", results_path)
    print(" -", archive)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--backbone",
        default="mobilenetv3_small",
        choices=SUPPORTED_BACKBONES,
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-dir", default=os.getcwd())
    parser.add_argument("--working-dir", default="/kaggle/working")
    parser.add_argument("--kaggle-input-root", default="/kaggle/input")
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument(
        "--sampler-checkpoint",
        default="gflownet_best_diverse.pth",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
