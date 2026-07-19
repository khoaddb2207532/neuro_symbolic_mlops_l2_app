"""Stage 0b - xoa anh do phan giai thap hon trong cac cap leakage.

Mac dinh chi tao ke hoach. Them ``--apply`` de xoa that:

    python -m pipelines.stage0b_remove_leakage --config params.yaml
    python -m pipelines.stage0b_remove_leakage --config params.yaml --apply
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.data.leakage_cleanup import (
    apply_deletion_plan,
    build_deletion_plan,
    count_images_by_split,
    write_deletion_csv,
    write_summary,
)
from src.utils.config import load_params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove lower-resolution images from cross-split duplicate leakage"
    )
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--duplicates-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--max-near-distance", type=int, default=1,
                        help="Remove near pairs with Hamming distance from 1 through this value")
    parser.add_argument("--tie-priority", nargs="+", default=("test", "val", "train"),
                        help="Tie-break when dimensions and split counts are tied")
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete files; without this flag only write a plan")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    params = load_params(args.config)
    data_dir = Path(args.data_dir or params["data_dir"])
    audit_dir = Path(args.output_dir or Path(params.get("output_dir", "outputs")) / "00_data_audit")
    duplicates_csv = Path(args.duplicates_csv or audit_dir / "duplicates.csv")

    plan, summary = build_deletion_plan(
        data_dir=data_dir,
        duplicates_csv=duplicates_csv,
        splits=args.splits,
        max_near_distance=args.max_near_distance,
        tie_priority=args.tie_priority,
    )
    plan_path = audit_dir / "leakage_deletion_plan.csv"
    write_deletion_csv(plan_path, plan)

    print(f"Split counts before: {summary['split_image_counts_before']}")
    print(f"Planned deletions: {summary['n_planned_deletions']}")
    print(f"Deletion plan: {plan_path.resolve()}")

    if not args.apply:
        summary["mode"] = "dry_run"
        write_summary(audit_dir / "leakage_cleanup_summary.json", summary)
        print("Dry run only. Review the plan, then rerun with --apply to delete files.")
        return 0

    results = apply_deletion_plan(data_dir, plan)
    deleted_path = audit_dir / "deleted_images.csv"
    write_deletion_csv(deleted_path, results)
    status_counts = Counter(item.status for item in results)
    summary["mode"] = "apply"
    summary["result_status_counts"] = dict(status_counts)
    summary["split_image_counts_after"] = count_images_by_split(data_dir, args.splits)
    write_summary(audit_dir / "leakage_cleanup_summary.json", summary)

    print(f"Deletion results: {dict(status_counts)}")
    print(f"Deleted-image report: {deleted_path.resolve()}")
    if status_counts.get("error", 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
