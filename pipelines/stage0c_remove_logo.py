"""Stage 0c - xoa logo co vi tri co dinh, xuat sang thu muc moi.

Anh mau 640x360 co logo THVL trong ROI xap xi x=522..634, y=13..54, tuong ung:
``--roi 0.815 0.035 0.99 0.15``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.data.logo_removal import process_logo_removal, write_logo_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove a fixed-position logo using a feathered neighboring background patch"
    )
    parser.add_argument("--input", required=True, help="One image or a class directory")
    parser.add_argument("--output-dir", required=True,
                        help="New directory; source images are never overwritten")
    parser.add_argument("--roi", nargs=4, type=float,
                        default=(0.815, 0.035, 0.99, 0.15),
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="Normalized logo box")
    parser.add_argument("--feather-radius", type=int, default=3)
    parser.add_argument("--neighbor-gap", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--report", default=None,
                        help="Default: <output-dir>/logo_removal_report.csv")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    records = process_logo_removal(
        input_path=Path(args.input), output_dir=output_dir,
        normalized_roi=args.roi, feather_radius=args.feather_radius,
        neighbor_gap=args.neighbor_gap, quality=args.jpeg_quality,
    )
    report_path = Path(args.report or output_dir / "logo_removal_report.csv")
    write_logo_report(report_path, records)
    statuses = Counter(record.status for record in records)
    print(f"Logo-removal results: {dict(statuses)}")
    print(f"Report: {report_path.resolve()}")
    return 2 if statuses.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
