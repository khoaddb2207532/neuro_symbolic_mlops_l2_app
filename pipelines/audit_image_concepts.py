# pyright: strict
"""Audit ``image_concepts.json`` TRƯỚC khi huấn luyện lại CBM.

Ví dụ::

    python -m pipelines.audit_image_concepts \\
        --input image_concepts.json \\
        --meta concept_annotations.json \\
        --dataset-dir /kaggle/input/.../khmer \\
        --out-dir outputs/concept_audit

``--input`` nhận cả các shard ``image_concepts_shard*.jsonl`` của notebook labeling
(khi đó bắt buộc có ``--meta`` để lấy tên concept). ``--meta`` cũng cung cấp prototype
``class_concepts`` giúp xác định concept nào "thuộc" lớp nào chính xác hơn so với suy từ dữ liệu.
Logic nằm ở ``src/data/concept_audit.py``; file này chỉ là CLI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from src.data.concept_audit import (
    AuditConfig,
    CoverageReport,
    analyze,
    check_dataset_coverage,
    format_summary,
    load_concept_table,
    write_report,
)


def build_parser() -> argparse.ArgumentParser:
    defaults = AuditConfig()
    parser = argparse.ArgumentParser(description="Audit per-image concept annotations cho CBM.")
    parser.add_argument("--input", nargs="+", type=Path, required=True,
                        help="image_concepts.json hoặc một/nhiều shard .jsonl")
    parser.add_argument("--meta", type=Path, default=None,
                        help="concept_annotations.json (concept_names, class_names, class_concepts)")
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Thư mục ảnh để kiểm tra nhãn có khớp path/tên file không")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/concept_audit"))
    parser.add_argument("--expected-per-image", type=float, default=None,
                        help="Số concept kỳ vọng mỗi ảnh (mặc định: suy từ home class)")
    parser.add_argument("--rare-freq", type=float, default=defaults.rare_freq)
    parser.add_argument("--common-freq", type=float, default=defaults.common_freq)
    parser.add_argument("--shared-rel", type=float, default=defaults.shared_rel)
    parser.add_argument("--max-foreign-rate", type=float, default=defaults.max_foreign_rate)
    parser.add_argument("--jaccard", type=float, default=defaults.jaccard_redundant)
    parser.add_argument("--top-pairs", type=int, default=defaults.top_pairs)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_paths: list[Path] = list(args.input)  # pyright: ignore[reportAny]
    meta_path: Path | None = args.meta  # pyright: ignore[reportAny]
    dataset_dir: Path | None = args.dataset_dir  # pyright: ignore[reportAny]
    out_dir: Path = args.out_dir  # pyright: ignore[reportAny]
    cfg = AuditConfig(
        rare_freq=float(args.rare_freq),  # pyright: ignore[reportAny]
        common_freq=float(args.common_freq),  # pyright: ignore[reportAny]
        shared_rel=float(args.shared_rel),  # pyright: ignore[reportAny]
        max_foreign_rate=float(args.max_foreign_rate),  # pyright: ignore[reportAny]
        jaccard_redundant=float(args.jaccard),  # pyright: ignore[reportAny]
        top_pairs=int(args.top_pairs),  # pyright: ignore[reportAny]
        expected_per_image=args.expected_per_image,  # pyright: ignore[reportAny]
    )

    table = load_concept_table(input_paths, meta_path)
    report = analyze(table, cfg)
    coverage: CoverageReport | None = None
    if dataset_dir is not None:
        coverage = check_dataset_coverage(table, dataset_dir)

    print(format_summary(report, coverage))
    written = write_report(out_dir, table, report, coverage)
    print("\nĐã ghi:")
    for path in written:
        print(f"  {path}")
    print("\nimage_stats.csv xếp từ ảnh có nhiều concept ngoài lớp nhất:")
    print("dùng làm danh sách ảnh cần kiểm lại ở bước verification.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
