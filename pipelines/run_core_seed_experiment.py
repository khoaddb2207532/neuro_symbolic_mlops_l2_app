"""Chạy hoặc tiếp tục thí nghiệm lõi MobileNetV3-Small + GFlowNet-DB.

Module này thay thế logic orchestration trong notebook Kaggle:

* cấu hình một seed;
* tùy chọn khôi phục artefact từ Kaggle Add Input;
* tự bỏ qua các stage đã hoàn tất;
* dùng matched budget bằng đúng số luật GFlowNet chọn;
* kiểm tra fairness và xuất CSV chuẩn hóa cho notebook tổng hợp.

Ví dụ:
    python -m pipelines.run_core_seed_experiment \
        --config params.yaml \
        --seed 42 \
        --data-dir /kaggle/input/.../vietnamese_cultural_dataset \
        --output-dir /kaggle/working/neuro_symbolic_mlops_l2_app/outputs_seed_42 \
        --kaggle-input-root /kaggle/input
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml
from openpyxl import load_workbook


BACKBONE = "mobilenetv3_small"
HEURISTICS = ("random", "topk_confidence", "greedy_coverage")


def _run_module(project: Path, config_path: Path, module: str, *args: object) -> None:
    command = [
        sys.executable,
        "-m",
        module,
        "--config",
        str(config_path),
        *map(str, args),
    ]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=project, check=True)


def _write_config(
    config_path: Path,
    *,
    seed: int,
    data_dir: Path,
    output_dir: Path,
    selection_budget: Optional[int] = None,
) -> Dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["seed"] = seed
    config["data_dir"] = str(data_dir)
    config["output_dir"] = str(output_dir)
    config["batch_size"] = 64
    config["num_workers"] = 2
    config["num_epochs"] = 100
    config["patience"] = 5
    config["baseline_comparison"]["selected_architecture"] = BACKBONE
    config["baseline_comparison"]["architectures"] = [BACKBONE]
    config["gflownet"]["loss_type"] = "db"
    if selection_budget is None:
        config.pop("selection_budget", None)
    else:
        config["selection_budget"] = int(selection_budget)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return config


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"Archive chứa path không an toàn: {member.name}")
        tar.extractall(destination)


def _find_restorable_output(input_root: Path, seed: int) -> Tuple[str, Path] | None:
    directories = sorted(
        path
        for path in input_root.rglob(f"outputs_seed_{seed}")
        if path.is_dir()
    )
    if directories:
        return "directory", directories[0]

    archives = sorted(input_root.rglob(f"seed_{seed}_artifacts.tar.gz"))
    if archives:
        return "archive", archives[0]
    return None


def _restore_if_available(input_root: Optional[Path], output_dir: Path, seed: int) -> None:
    if input_root is None or not input_root.is_dir():
        return
    source = _find_restorable_output(input_root, seed)
    if source is None:
        print(
            f"Không tìm thấy output seed {seed} trong {input_root}; "
            "bắt đầu/chạy tiếp bằng artefact hiện có.",
            flush=True,
        )
        return

    kind, path = source
    print(f"Khôi phục output seed {seed} từ {path}", flush=True)
    if kind == "directory":
        shutil.copytree(path, output_dir, dirs_exist_ok=True)
    else:
        _safe_extract_tar(path, output_dir)


def _xlsx_row_count(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.active
        # File save_rules_excel có một hàng header.
        return max(sheet.max_row - 1, 0)
    finally:
        workbook.close()


def _first_report(directory: Path) -> Optional[Path]:
    reports = sorted(directory.glob("*classification_report.txt"))
    return reports[0] if reports else None


def _report_metrics(path: Path) -> Tuple[float, float]:
    text = path.read_text(encoding="utf-8")
    accuracy_match = re.search(r"Overall Accuracy:\s*([0-9.]+)%", text)
    if accuracy_match is None:
        raise ValueError(f"Không đọc được accuracy từ {path}")
    macro_line = next(
        (line for line in text.splitlines() if line.strip().startswith("macro avg")),
        None,
    )
    if macro_line is None:
        raise ValueError(f"Không đọc được macro-F1 từ {path}")
    return float(accuracy_match.group(1)) / 100.0, float(macro_line.split()[4])


def _required_dataset_splits(data_dir: Path) -> None:
    missing = [name for name in ("train", "val", "test") if not (data_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"{data_dir} thiếu các split: {missing}")


def _write_csv(path: Path, rows: Iterable[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    project = Path(args.project_dir).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project / config_path
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    working_dir = Path(args.working_dir)
    input_root = Path(args.kaggle_input_root) if args.kaggle_input_root else None

    _required_dataset_splits(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _restore_if_available(input_root, output_dir, args.seed)
    config = _write_config(
        config_path,
        seed=args.seed,
        data_dir=data_dir,
        output_dir=output_dir,
    )

    baseline_dir = output_dir / "baseline_comparison" / BACKBONE
    baseline_checkpoint = baseline_dir / "baseline_best.pth"
    baseline_report = _first_report(baseline_dir)
    if baseline_checkpoint.exists() and baseline_report is not None:
        print("SKIP baseline:", baseline_checkpoint)
    else:
        print(
            "RESUME baseline:",
            f"checkpoint={'OK' if baseline_checkpoint.exists() else 'MISSING'},",
            f"report={'OK' if baseline_report is not None else 'MISSING'}",
        )
        _run_module(
            project,
            config_path,
            "pipelines.train_all_baselines",
            "--models",
            BACKBONE,
            "--epochs",
            config["num_epochs"],
        )
        baseline_report = _first_report(baseline_dir)
        if not baseline_checkpoint.exists() or baseline_report is None:
            raise FileNotFoundError(
                "Baseline chạy xong nhưng thiếu checkpoint/report."
            )

    features_dir = output_dir / "02_features"
    required_features = [
        features_dir / f"{split}_{kind}.pt"
        for split in ("train", "val", "test")
        for kind in ("features", "labels")
    ]
    missing_features = [path for path in required_features if not path.exists()]
    if not missing_features:
        print("SKIP Stage 2: feature artefact đã tồn tại.")
    else:
        print(
            "RESUME Stage 2, thiếu:",
            ", ".join(path.name for path in missing_features),
        )
        _run_module(project, config_path, "pipelines.stage2_extract_features")
        missing_features = [path for path in required_features if not path.exists()]
        if missing_features:
            raise FileNotFoundError(
                "Stage 2 chạy xong nhưng còn thiếu: "
                + ", ".join(map(str, missing_features))
            )

    raw_rules = output_dir / "03_rules" / "raw_rules.pkl"
    if raw_rules.exists():
        print("SKIP Stage 3:", raw_rules)
    else:
        _run_module(project, config_path, "pipelines.stage3_extract_rules")

    gfn_rules_dir = output_dir / "04_filtered_rules"
    gfn_rules_xlsx = gfn_rules_dir / "selected_rules.xlsx"
    gfn_rules_pickle = gfn_rules_dir / "selected_rules.pkl"
    if gfn_rules_xlsx.exists() and gfn_rules_pickle.exists():
        print("SKIP Stage 4:", gfn_rules_xlsx)
    else:
        print(
            "RESUME Stage 4:",
            f"xlsx={'OK' if gfn_rules_xlsx.exists() else 'MISSING'},",
            f"pickle={'OK' if gfn_rules_pickle.exists() else 'MISSING'}",
        )
        _run_module(project, config_path, "pipelines.stage4_select_rules_gflownet")
        if not gfn_rules_xlsx.exists() or not gfn_rules_pickle.exists():
            raise FileNotFoundError(
                "Stage 4 chạy xong nhưng thiếu selected_rules.xlsx/.pkl."
            )

    budget = _xlsx_row_count(gfn_rules_xlsx)
    if budget <= 0:
        raise RuntimeError("GFlowNet không chọn luật nào.")
    print(f"Matched budget K={budget}", flush=True)
    _write_config(
        config_path,
        seed=args.seed,
        data_dir=data_dir,
        output_dir=output_dir,
        selection_budget=budget,
    )

    gfn_report = _first_report(output_dir / "05_rules_model")
    if gfn_report is not None:
        print("SKIP Stage 5 GFlowNet:", gfn_report)
    else:
        _run_module(project, config_path, "pipelines.stage5_train_rule_regularized")
        gfn_report = _first_report(output_dir / "05_rules_model")
        if gfn_report is None:
            raise FileNotFoundError("Stage 5 hoàn tất nhưng không tạo classification report.")

    # Resume chi tiết từng heuristic. Script heuristic ghi artefact của mỗi
    # phương pháp vào thư mục riêng, vì vậy một method hoàn tất không cần chạy
    # lại khi method khác bị gián đoạn.
    heuristic_reports: Dict[str, Path] = {}
    for method in HEURISTICS:
        method_rules = (
            output_dir
            / f"04_filtered_rules_{method}"
            / "selected_rules.xlsx"
        )
        method_report = _first_report(
            output_dir / f"05_rules_model_{method}"
        )
        if method_rules.exists() and method_report is not None:
            print(f"SKIP heuristic {method}: {method_report}")
            heuristic_reports[method] = method_report
            continue

        print(
            f"RESUME heuristic {method}: "
            f"rules={'OK' if method_rules.exists() else 'MISSING'}, "
            f"report={'OK' if method_report is not None else 'MISSING'}"
        )
        _run_module(
            project,
            config_path,
            "pipelines.stage5_train_rule_regularized_heuristics",
            "--methods",
            method,
            "--random_seed",
            args.seed,
        )
        method_report = _first_report(
            output_dir / f"05_rules_model_{method}"
        )
        if not method_rules.exists() or method_report is None:
            raise FileNotFoundError(
                f"Heuristic {method} chạy xong nhưng thiếu rules/report."
            )
        heuristic_reports[method] = method_report

    counts = {"gflownet": budget}
    for method in HEURISTICS:
        counts[method] = _xlsx_row_count(
            output_dir / f"04_filtered_rules_{method}" / "selected_rules.xlsx"
        )
        if counts[method] != budget:
            raise RuntimeError(
                f"Vi phạm matched-budget seed {args.seed}: "
                f"{method}={counts[method]}, GFlowNet={budget}."
            )

    fairness_path = working_dir / f"seed_{args.seed}_fairness.csv"
    _write_csv(
        fairness_path,
        [{"seed": args.seed, **counts, "matched_budget_pass": True}],
        [
            "seed",
            "gflownet",
            "random",
            "topk_confidence",
            "greedy_coverage",
            "matched_budget_pass",
        ],
    )

    baseline_accuracy, baseline_f1 = _report_metrics(baseline_report)
    gfn_accuracy, gfn_f1 = _report_metrics(gfn_report)

    result_rows = [
        {
            "seed": args.seed,
            "method": "cnn_baseline",
            "n_rules_selected": 0,
            "test_accuracy": baseline_accuracy,
            "test_f1_macro": baseline_f1,
        },
        {
            "seed": args.seed,
            "method": "gflownet_db",
            "n_rules_selected": budget,
            "test_accuracy": gfn_accuracy,
            "test_f1_macro": gfn_f1,
        },
    ]
    heuristic_comparison_rows = []
    for method in HEURISTICS:
        accuracy, f1_macro = _report_metrics(heuristic_reports[method])
        result_rows.append(
            {
                "seed": args.seed,
                "method": method,
                "n_rules_selected": counts[method],
                "test_accuracy": accuracy,
                "test_f1_macro": f1_macro,
            }
        )
        heuristic_comparison_rows.append(
            {
                "method": method,
                "n_rules_selected": counts[method],
                "test_accuracy": accuracy,
                "test_f1_macro": f1_macro,
                "save_dir": str(output_dir / f"05_rules_model_{method}"),
            }
        )

    # Mỗi lần gọi script cho một method sẽ ghi đè CSV tạm. Dựng lại bảng
    # canonical từ ba report riêng sau khi tất cả method đã hoàn tất.
    heuristic_csv = output_dir / "rule_selection_finetune_comparison.csv"
    _write_csv(
        heuristic_csv,
        heuristic_comparison_rows,
        [
            "method",
            "n_rules_selected",
            "test_accuracy",
            "test_f1_macro",
            "save_dir",
        ],
    )

    results_path = working_dir / f"seed_{args.seed}_results.csv"
    _write_csv(
        results_path,
        result_rows,
        [
            "seed",
            "method",
            "n_rules_selected",
            "test_accuracy",
            "test_f1_macro",
        ],
    )
    archive = shutil.make_archive(
        str(working_dir / f"seed_{args.seed}_artifacts"),
        "gztar",
        root_dir=output_dir,
    )
    print("\nHoàn tất:")
    print(" -", results_path)
    print(" -", fairness_path)
    print(" -", archive)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-dir", default=os.getcwd())
    parser.add_argument("--working-dir", default="/kaggle/working")
    parser.add_argument(
        "--kaggle-input-root",
        default="/kaggle/input",
        help="Thư mục Add Input. Module tự restore output cùng seed nếu tìm thấy.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
