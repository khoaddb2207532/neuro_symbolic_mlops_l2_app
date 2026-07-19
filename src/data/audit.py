"""Kiem toan du lieu anh truoc khi huan luyen.

Module nay khong tao hoac thay doi split. No chi doc cau truc
``data_dir/<split>/<class>/*`` va tao cac bang thong ke de phat hien:

* mat can bang lop va lop khong dong nhat giua cac split;
* anh hong, kich thuoc bat thuong va cac thong ke anh co ban;
* anh trung byte, anh gan trung va leakage giua cac split;
* group leakage neu ten tep ma hoa id doi tuong/phien chup.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageStat

from src.data.dataset import VALID_EXTENSIONS


@dataclass
class ImageRecord:
    path: str
    split: str
    class_name: str
    size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    aspect_ratio: Optional[float] = None
    mode: Optional[str] = None
    brightness: Optional[float] = None
    sha256: Optional[str] = None
    dhash: Optional[str] = None
    group_id: Optional[str] = None
    error: Optional[str] = None


class BKTree:
    """BK-tree nho gon cho hash 64 bit voi khoang cach Hamming."""

    def __init__(self) -> None:
        self.root: Optional[Tuple[int, Dict[int, tuple]]] = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            current, children = node
            distance = self.distance(value, current)
            if distance == 0:
                return
            child = children.get(distance)
            if child is None:
                children[distance] = (value, {})
                return
            node = child

    def query(self, value: int, max_distance: int) -> Iterable[Tuple[int, int]]:
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            current, children = stack.pop()
            distance = self.distance(value, current)
            if distance <= max_distance:
                yield current, distance
            low, high = distance - max_distance, distance + max_distance
            stack.extend(child for edge, child in children.items() if low <= edge <= high)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash(image: Image.Image) -> str:
    """Tinh difference hash 64 bit, doc lap voi thu vien imagehash."""
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _extract_group_id(path: Path, pattern: Optional[re.Pattern[str]]) -> Optional[str]:
    if pattern is None:
        return None
    match = pattern.search(path.stem)
    if match is None:
        return None
    if "group" in match.groupdict():
        return match.group("group")
    if match.groups():
        return match.group(1)
    return match.group(0)


def inspect_image(path: Path, data_dir: Path, split: str, class_name: str,
                  group_pattern: Optional[re.Pattern[str]] = None) -> ImageRecord:
    record = ImageRecord(
        path=path.relative_to(data_dir).as_posix(),
        split=split,
        class_name=class_name,
        size_bytes=path.stat().st_size,
        group_id=_extract_group_id(path, group_pattern),
    )
    try:
        record.sha256 = _sha256(path)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            record.width, record.height = image.size
            record.aspect_ratio = record.width / record.height if record.height else None
            record.mode = image.mode
            record.brightness = float(ImageStat.Stat(image.convert("L")).mean[0])
            record.dhash = _dhash(image)
    except Exception as exc:  # PIL co nhieu loai loi theo dinh dang anh
        record.error = f"{type(exc).__name__}: {exc}"
    return record


def _describe(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {key: None for key in ("min", "q25", "median", "mean", "q75", "max")}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def _duplicate_groups(records: Sequence[ImageRecord], field: str) -> List[List[ImageRecord]]:
    groups: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        value = getattr(record, field)
        if value:
            groups[value].append(record)
    return [group for group in groups.values() if len(group) > 1]


def _near_duplicate_pairs(records: Sequence[ImageRecord], max_distance: int,
                          max_pairs: int) -> Tuple[List[dict], bool]:
    """Tra ve pair gan trung; pair trung byte duoc loai khoi ket qua."""
    by_hash: Dict[int, List[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.dhash:
            by_hash[int(record.dhash, 16)].append(record)

    tree = BKTree()
    pairs: List[dict] = []
    truncated = False
    for hash_value, current_records in by_hash.items():
        for other_hash, distance in tree.query(hash_value, max_distance):
            for left in by_hash[other_hash]:
                for right in current_records:
                    if left.sha256 == right.sha256:
                        continue
                    pairs.append({
                        "path_a": left.path,
                        "split_a": left.split,
                        "class_a": left.class_name,
                        "path_b": right.path,
                        "split_b": right.split,
                        "class_b": right.class_name,
                        "hamming_distance": distance,
                        "cross_split": left.split != right.split,
                        "cross_class": left.class_name != right.class_name,
                    })
                    if len(pairs) >= max_pairs:
                        truncated = True
                        return pairs, truncated
        tree.add(hash_value)
    return pairs, truncated


def _protocol_fingerprint(records: Sequence[ImageRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.path):
        line = f"{record.path}\t{record.sha256 or 'ERROR'}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def audit_dataset(data_dir: Path, splits: Sequence[str] = ("train", "val", "test"),
                  near_duplicate_distance: int = 4, max_near_duplicate_pairs: int = 10000,
                  group_regex: Optional[str] = None) -> Tuple[List[ImageRecord], dict, List[dict]]:
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not 0 <= near_duplicate_distance <= 64:
        raise ValueError("near_duplicate_distance must be between 0 and 64")
    group_pattern = re.compile(group_regex) if group_regex else None

    records: List[ImageRecord] = []
    split_classes: Dict[str, List[str]] = {}
    missing_splits: List[str] = []
    ignored_non_images = 0
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            missing_splits.append(split)
            split_classes[split] = []
            continue
        classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        split_classes[split] = classes
        for class_name in classes:
            for path in sorted((split_dir / class_name).iterdir()):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in VALID_EXTENSIONS:
                    ignored_non_images += 1
                    continue
                records.append(inspect_image(path, data_dir, split, class_name, group_pattern))

    class_counts = Counter((record.split, record.class_name) for record in records)
    by_split: Dict[str, dict] = {}
    for split in splits:
        counts = {name: class_counts[(split, name)] for name in split_classes[split]}
        nonzero = [count for count in counts.values() if count > 0]
        by_split[split] = {
            "n_images": sum(counts.values()),
            "n_classes": len(counts),
            "class_counts": counts,
            "imbalance_ratio_max_over_min": (
                max(nonzero) / min(nonzero) if nonzero else None
            ),
        }

    healthy = [record for record in records if record.error is None]
    exact_groups = _duplicate_groups(healthy, "sha256")
    exact_rows: List[dict] = []
    for group_id, group in enumerate(exact_groups, start=1):
        for record in group:
            exact_rows.append({
                "duplicate_type": "exact",
                "group_id": group_id,
                "path": record.path,
                "split": record.split,
                "class_name": record.class_name,
                "sha256": record.sha256,
                "cross_split": len({item.split for item in group}) > 1,
                "cross_class": len({item.class_name for item in group}) > 1,
            })

    near_pairs, near_truncated = _near_duplicate_pairs(
        healthy, near_duplicate_distance, max_near_duplicate_pairs
    )
    group_leakage: List[dict] = []
    if group_pattern:
        groups: Dict[str, List[ImageRecord]] = defaultdict(list)
        for record in records:
            if record.group_id:
                groups[record.group_id].append(record)
        for group_id, group in groups.items():
            present_splits = sorted({record.split for record in group})
            if len(present_splits) > 1:
                group_leakage.append({
                    "group_id": group_id,
                    "splits": present_splits,
                    "paths": [record.path for record in group],
                })

    class_sets = {split: set(names) for split, names in split_classes.items()}
    union_classes = sorted(set().union(*class_sets.values()) if class_sets else set())
    class_mismatch = {
        split: sorted(set(union_classes) - classes)
        for split, classes in class_sets.items()
        if set(union_classes) != classes
    }
    summary = {
        "data_dir": str(data_dir),
        "splits": list(splits),
        "protocol_fingerprint_sha256": _protocol_fingerprint(records),
        "n_images": len(records),
        "n_healthy_images": len(healthy),
        "n_corrupt_images": len(records) - len(healthy),
        "n_ignored_non_images": ignored_non_images,
        "missing_splits": missing_splits,
        "classes": union_classes,
        "class_mismatch": class_mismatch,
        "by_split": by_split,
        "image_statistics": {
            "width": _describe([record.width for record in healthy if record.width is not None]),
            "height": _describe([record.height for record in healthy if record.height is not None]),
            "aspect_ratio": _describe([record.aspect_ratio for record in healthy if record.aspect_ratio is not None]),
            "brightness_0_255": _describe([record.brightness for record in healthy if record.brightness is not None]),
            "size_bytes": _describe([record.size_bytes for record in healthy]),
            "modes": dict(Counter(record.mode for record in healthy)),
        },
        "duplicates": {
            "n_exact_groups": len(exact_groups),
            "n_exact_images": sum(len(group) for group in exact_groups),
            "n_exact_cross_split_groups": sum(len({item.split for item in group}) > 1 for group in exact_groups),
            "n_exact_cross_class_groups": sum(len({item.class_name for item in group}) > 1 for group in exact_groups),
            "near_duplicate_hamming_threshold": near_duplicate_distance,
            "n_near_duplicate_pairs": len(near_pairs),
            "n_near_cross_split_pairs": sum(pair["cross_split"] for pair in near_pairs),
            "n_near_cross_class_pairs": sum(pair["cross_class"] for pair in near_pairs),
            "near_pairs_truncated": near_truncated,
        },
        "groups": {
            "regex": group_regex,
            "n_cross_split_groups": len(group_leakage),
            "cross_split_groups": group_leakage,
        },
    }
    return records, summary, exact_rows + [dict(duplicate_type="near", **pair) for pair in near_pairs]


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_audit_report(output_dir: Path, records: Sequence[ImageRecord], summary: dict,
                       duplicate_rows: Sequence[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    record_rows = [asdict(record) for record in records]
    _write_csv(output_dir / "image_manifest.csv", record_rows, list(ImageRecord.__dataclass_fields__))

    class_rows = []
    for split, split_summary in summary["by_split"].items():
        total = split_summary["n_images"]
        for class_name, count in split_summary["class_counts"].items():
            class_rows.append({
                "split": split,
                "class_name": class_name,
                "count": count,
                "proportion": count / total if total else 0.0,
            })
    _write_csv(output_dir / "class_distribution.csv", class_rows,
               ("split", "class_name", "count", "proportion"))

    duplicate_fields = (
        "duplicate_type", "group_id", "path", "split", "class_name", "sha256",
        "path_a", "split_a", "class_a", "path_b", "split_b", "class_b",
        "hamming_distance", "cross_split", "cross_class",
    )
    _write_csv(output_dir / "duplicates.csv", duplicate_rows, duplicate_fields)


def protocol_violations(summary: dict, fail_on_corrupt: bool = False,
                        fail_on_leakage: bool = False,
                        fail_on_class_mismatch: bool = False) -> List[str]:
    violations = []
    if summary["missing_splits"]:
        violations.append(f"missing splits: {summary['missing_splits']}")
    if fail_on_corrupt and summary["n_corrupt_images"]:
        violations.append(f"corrupt images: {summary['n_corrupt_images']}")
    if fail_on_class_mismatch and summary["class_mismatch"]:
        violations.append(f"class mismatch: {summary['class_mismatch']}")
    if fail_on_leakage:
        duplicates = summary["duplicates"]
        if duplicates["n_exact_cross_split_groups"]:
            violations.append(
                f"exact duplicate groups across splits: {duplicates['n_exact_cross_split_groups']}"
            )
        if duplicates["n_near_cross_split_pairs"]:
            violations.append(
                f"near-duplicate pairs across splits: {duplicates['n_near_cross_split_pairs']}"
            )
        if summary["groups"]["n_cross_split_groups"]:
            violations.append(
                f"semantic groups across splits: {summary['groups']['n_cross_split_groups']}"
            )
    return violations
