"""Print classification summaries and selected rules for a dual-GPU CBM run."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from pipelines.run_cbm_neuro_symbolic_pipeline import (
    CONTROL_CONCEPT_FREE_SUBDIR,
    CONTROL_ZERO_RULE_WEIGHT_SUBDIR,
    EXPECTED_MODELS,
)

CONTROL_SUBDIRS = {CONTROL_ZERO_RULE_WEIGHT_SUBDIR, CONTROL_CONCEPT_FREE_SUBDIR}


def print_classification_reports(seeds: list[int], root: Path) -> None:
    summary_csv = root / "dual_gpu_all_seeds_classification_summary.csv"
    if summary_csv.exists():
        print("=== CLASSIFICATION REPORT SUMMARY (PRECISION, RECALL, F1) ===")
        print(pd.read_csv(summary_csv).to_string(index=False))
    else:
        print("Reading classification summaries from individual seed folders...")
        summaries = []
        for seed in seeds:
            summary_path = root / f"outputs_seed_{seed}" / "seed_classification_summary.csv"
            if summary_path.exists():
                summaries.append(pd.read_csv(summary_path))
        if summaries:
            print(pd.concat(summaries, ignore_index=True).to_string(index=False))
        else:
            print("No classification summary CSV found.")

    print("\nSaved report files:")
    for seed in seeds:
        seed_dir = root / f"outputs_seed_{seed}"
        print(f"\n>>> SEED {seed}:")
        for report in sorted(seed_dir.glob("**/*classification_report*")):
            print(f"  - {report}")
        for confusion_matrix in sorted(seed_dir.glob("**/*confusion_matrix.png")):
            print(f"  - {confusion_matrix}")


def expected_model_labels(include_controls: bool) -> list[str]:
    """Nhãn mô hình phải có trong bảng: 5 nếu có đối chứng, 3 nếu chạy với --skip-controls."""
    return [
        label for label, subdir in EXPECTED_MODELS
        if include_controls or subdir not in CONTROL_SUBDIRS
    ]


def verify_all_models(seeds: list[int], root: Path, include_controls: bool = True) -> list[str]:
    """Trả về danh sách lỗi; rỗng nghĩa là mỗi seed có đủ mô hình đã khai báo."""
    expected = expected_model_labels(include_controls)
    problems: list[str] = []
    for seed in seeds:
        summary_path = root / f"outputs_seed_{seed}" / "seed_classification_summary.csv"
        if not summary_path.exists():
            problems.append(f"Seed {seed}: không có {summary_path.name} (pipeline chưa tới bước tổng hợp).")
            continue
        got = set(pd.read_csv(summary_path)["Model"].astype(str))
        missing = [label for label in expected if label not in got]
        if missing:
            problems.append(
                f"Seed {seed}: thiếu {len(missing)}/{len(expected)} mô hình: {', '.join(missing)}"
            )

    return problems


def print_top_rules(seed: int, root: Path) -> None:
    rules_path = root / f"outputs_seed_{seed}" / "04_filtered_rules" / "selected_rules.xlsx"
    if not rules_path.exists():
        print("No selected_rules.xlsx found.")
        return

    rules = pd.read_excel(rules_path)
    print(f"\n=== TOP CONCEPT RULES SELECTED BY GFLOWNET (SEED {seed}) ===")
    print(rules.head(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs=2, type=int, required=True)
    parser.add_argument("--top-rules-seed", type=int)
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Khai báo đã chạy với --skip-controls: chỉ kiểm tra 3 mô hình chính",
    )
    args = parser.parse_args()

    root = Path.cwd()
    if args.top_rules_seed is not None:
        print_top_rules(args.top_rules_seed, root)
        return

    print_classification_reports(args.seeds, root)
    include_controls = not args.skip_controls
    problems = verify_all_models(args.seeds, root, include_controls=include_controls)
    n_expected = len(expected_model_labels(include_controls))
    if problems:
        print(f"\nKHÔNG ĐỦ {n_expected} MÔ HÌNH:", file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        sys.exit(1)
    print(f"\nOK: mỗi seed có đủ {n_expected} mô hình.")


if __name__ == "__main__":
    main()
