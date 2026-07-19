"""Stage 0 - kiem toan va khoa giao thuc du lieu truoc khi train.

Vi du:
    python -m pipelines.stage0_audit_data --config params.yaml --strict

Neu ten anh co dang ``artifact123_view2.jpg``, co the kiem tra group leakage:
    python -m pipelines.stage0_audit_data --group-regex "^(?P<group>[^_]+)" --strict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.audit import audit_dataset, protocol_violations, write_audit_report
from src.utils.config import load_params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit image dataset before model training")
    parser.add_argument("--config", default="params.yaml", help="YAML config containing data_dir/output_dir")
    parser.add_argument("--data-dir", default=None, help="Override data_dir from config")
    parser.add_argument("--output-dir", default=None, help="Default: <output_dir>/00_data_audit")
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--near-duplicate-distance", type=int, default=4,
                        help="Maximum 64-bit dHash Hamming distance")
    parser.add_argument("--max-near-duplicate-pairs", type=int, default=10000)
    parser.add_argument("--group-regex", default=None,
                        help="Regex over filename stem; use named group (?P<group>...) or first group")
    parser.add_argument("--fail-on-corrupt", action="store_true")
    parser.add_argument("--fail-on-leakage", action="store_true")
    parser.add_argument("--fail-on-class-mismatch", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="Enable all fail-on policies")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    params = load_params(args.config)
    data_dir = Path(args.data_dir or params["data_dir"])
    output_dir = Path(args.output_dir or Path(params.get("output_dir", "outputs")) / "00_data_audit")

    records, summary, duplicates = audit_dataset(
        data_dir=data_dir,
        splits=args.splits,
        near_duplicate_distance=args.near_duplicate_distance,
        max_near_duplicate_pairs=args.max_near_duplicate_pairs,
        group_regex=args.group_regex,
    )
    write_audit_report(output_dir, records, summary, duplicates)

    strict = args.strict
    violations = protocol_violations(
        summary,
        fail_on_corrupt=strict or args.fail_on_corrupt,
        fail_on_leakage=strict or args.fail_on_leakage,
        fail_on_class_mismatch=strict or args.fail_on_class_mismatch,
    )

    print(f"Audited {summary['n_images']} images; report: {output_dir.resolve()}")
    print(f"Protocol fingerprint: {summary['protocol_fingerprint_sha256']}")
    print(f"Corrupt images: {summary['n_corrupt_images']}")
    print(f"Exact cross-split duplicate groups: {summary['duplicates']['n_exact_cross_split_groups']}")
    print(f"Near cross-split duplicate pairs: {summary['duplicates']['n_near_cross_split_pairs']}")
    if violations:
        print("Protocol rejected:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 2
    print("Protocol accepted under the enabled policies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
