"""Print classification summaries and selected rules for a dual-GPU CBM run."""

import argparse
from pathlib import Path

import pandas as pd


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
    args = parser.parse_args()

    root = Path.cwd()
    if args.top_rules_seed is not None:
        print_top_rules(args.top_rules_seed, root)
    else:
        print_classification_reports(args.seeds, root)


if __name__ == "__main__":
    main()
