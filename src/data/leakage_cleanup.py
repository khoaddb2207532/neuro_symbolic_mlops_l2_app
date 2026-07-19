"""Lap ke hoach va thuc thi xoa anh leakage dua tren bao cao Stage 0.

Chi hai loai canh duoc xem la du dieu kien:

* anh trung chinh xac (SHA-256) nam o nhieu split;
* anh gan trung nam o nhieu split va co dHash Hamming distance dung bang nguong
  yeu cau (mac dinh la 1).

Nhung anh lien quan duoc gom thanh thanh phan lien thong. Trong moi thanh phan,
split chua anh co do phan giai cao nhat duoc giu; anh thuoc split khac bi lap
ke hoach xoa. Neu do phan giai bang nhau (thuong gap voi exact duplicate),
kich thuoc split va tie-priority duoc dung lam tie-break.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageOps

from src.data.dataset import VALID_EXTENSIONS


TRUE_VALUES = {"1", "true", "yes", "y"}


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left

    def components(self) -> List[Set[str]]:
        groups: Dict[str, Set[str]] = defaultdict(set)
        for item in self.parent:
            groups[self.find(item)].add(item)
        return list(groups.values())


@dataclass
class DeletionItem:
    component_id: int
    path: str
    split: str
    class_name: str
    keep_split: str
    split_image_count: int
    keep_split_image_count: int
    width: Optional[int]
    height: Optional[int]
    pixel_area: Optional[int]
    keep_width: Optional[int]
    keep_height: Optional[int]
    keep_pixel_area: Optional[int]
    representative_keep_path: str
    selection_rule: str
    reason_types: str
    related_paths: str
    sha256_before: Optional[str]
    action: str = "delete"
    status: str = "planned"
    error: str = ""
    processed_at_utc: str = ""


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def _normalise_relative_path(value: str) -> str:
    return Path(value.strip().replace("\\", "/")).as_posix()


def _safe_resolve(data_dir: Path, relative_path: str) -> Path:
    candidate = (data_dir / relative_path).resolve()
    root = data_dir.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Unsafe path outside data_dir: {relative_path}") from exc
    return candidate


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_dimensions(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if not path.is_file():
        return None, None
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
        return int(width), int(height)
    except Exception:
        return None, None


def _quality_key(width: Optional[int], height: Optional[int]) -> Tuple[int, int, int]:
    """Uu tien dien tich pixel, roi canh ngan, sau cung canh dai."""
    if width is None or height is None:
        return -1, -1, -1
    return width * height, min(width, height), max(width, height)


def count_images_by_split(data_dir: Path, splits: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            counts[split] = 0
            continue
        counts[split] = sum(
            1
            for path in split_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )
    return counts


def _read_selected_edges(duplicates_csv: Path, max_near_distance: int) -> Tuple[
    UnionFind, Dict[str, Set[str]], Dict[str, Tuple[str, str]]
]:
    union_find = UnionFind()
    path_reasons: Dict[str, Set[str]] = defaultdict(set)
    metadata: Dict[str, Tuple[str, str]] = {}
    exact_groups: Dict[str, List[str]] = defaultdict(list)

    with duplicates_csv.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            duplicate_type = (row.get("duplicate_type") or "").strip().lower()
            if not _as_bool(row.get("cross_split")):
                continue

            if duplicate_type == "exact":
                path = _normalise_relative_path(row.get("path") or "")
                group_id = (row.get("group_id") or "").strip()
                if not path or not group_id:
                    continue
                exact_groups[group_id].append(path)
                metadata[path] = ((row.get("split") or "").strip(),
                                  (row.get("class_name") or "").strip())
                path_reasons[path].add("exact_cross_split")

            elif duplicate_type == "near":
                try:
                    distance = int(float(row.get("hamming_distance") or "-1"))
                except ValueError:
                    continue
                if distance < 1 or distance > max_near_distance:
                    continue
                left = _normalise_relative_path(row.get("path_a") or "")
                right = _normalise_relative_path(row.get("path_b") or "")
                if not left or not right:
                    continue
                union_find.union(left, right)
                metadata[left] = ((row.get("split_a") or "").strip(),
                                  (row.get("class_a") or "").strip())
                metadata[right] = ((row.get("split_b") or "").strip(),
                                   (row.get("class_b") or "").strip())
                path_reasons[left].add(f"near_cross_split_dhash_{distance}")
                path_reasons[right].add(f"near_cross_split_dhash_{distance}")

    for paths in exact_groups.values():
        if not paths:
            continue
        first = paths[0]
        union_find.add(first)
        for path in paths[1:]:
            union_find.union(first, path)

    return union_find, path_reasons, metadata


def build_deletion_plan(
    data_dir: Path,
    duplicates_csv: Path,
    splits: Sequence[str] = ("train", "val", "test"),
    max_near_distance: int = 1,
    tie_priority: Sequence[str] = ("test", "val", "train"),
) -> Tuple[List[DeletionItem], dict]:
    """Tao plan; khong xoa tep.

    Split giu lai la split chua anh co quality key lon nhat trong component.
    Neu nhieu split co cung quality key cao nhat, chon theo (tong so anh tang
    dan, tie priority). Chi cac split trong component tham gia quyet dinh.
    """
    data_dir = data_dir.resolve()
    duplicates_csv = duplicates_csv.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not duplicates_csv.is_file():
        raise FileNotFoundError(f"Duplicate report not found: {duplicates_csv}")
    if not 1 <= max_near_distance <= 64:
        raise ValueError("max_near_distance must be between 1 and 64")

    split_counts = count_images_by_split(data_dir, splits)
    priority = {split: index for index, split in enumerate(tie_priority)}
    union_find, path_reasons, metadata = _read_selected_edges(
        duplicates_csv, max_near_distance
    )

    plan: List[DeletionItem] = []
    components_summary: List[dict] = []
    sorted_components = sorted(
        union_find.components(), key=lambda component: sorted(component)[0]
    )
    for component_id, component in enumerate(sorted_components, start=1):
        component_splits = {metadata[path][0] for path in component if path in metadata}
        component_splits.discard("")
        if len(component_splits) < 2:
            continue
        unknown_splits = component_splits - set(split_counts)
        if unknown_splits:
            raise ValueError(f"Unknown splits in duplicate report: {sorted(unknown_splits)}")

        paths_sorted = sorted(component)
        dimensions = {
            path: _image_dimensions(_safe_resolve(data_dir, path))
            for path in paths_sorted
        }
        best_quality_by_split = {
            split: max(
                (
                    _quality_key(*dimensions[path])
                    for path in paths_sorted
                    if metadata[path][0] == split
                ),
                default=(-1, -1, -1),
            )
            for split in component_splits
        }
        best_quality = max(best_quality_by_split.values())
        quality_winners = {
            split
            for split, quality in best_quality_by_split.items()
            if quality == best_quality
        }
        keep_split = min(
            quality_winners,
            key=lambda split: (
                split_counts[split], priority.get(split, len(priority)), split
            ),
        )
        keep_candidates = [
            path
            for path in paths_sorted
            if metadata[path][0] == keep_split
            and _quality_key(*dimensions[path]) == best_quality
        ]
        representative_keep_path = min(keep_candidates)
        keep_width, keep_height = dimensions[representative_keep_path]
        keep_area = (
            keep_width * keep_height
            if keep_width is not None and keep_height is not None
            else None
        )
        selection_rule = (
            "highest_resolution"
            if len(quality_winners) == 1
            else "resolution_tie_then_smaller_split"
        )
        component_reasons = sorted(
            set().union(*(path_reasons.get(path, set()) for path in component))
        )
        delete_paths = []
        keep_paths = []
        for relative_path in paths_sorted:
            split, class_name = metadata[relative_path]
            resolved = _safe_resolve(data_dir, relative_path)
            width, height = dimensions[relative_path]
            pixel_area = (
                width * height
                if width is not None and height is not None
                else None
            )
            if split == keep_split:
                keep_paths.append(relative_path)
                continue
            delete_paths.append(relative_path)
            plan.append(DeletionItem(
                component_id=component_id,
                path=relative_path,
                split=split,
                class_name=class_name,
                keep_split=keep_split,
                split_image_count=split_counts[split],
                keep_split_image_count=split_counts[keep_split],
                width=width,
                height=height,
                pixel_area=pixel_area,
                keep_width=keep_width,
                keep_height=keep_height,
                keep_pixel_area=keep_area,
                representative_keep_path=representative_keep_path,
                selection_rule=selection_rule,
                reason_types=";".join(component_reasons),
                related_paths=";".join(path for path in paths_sorted if path != relative_path),
                sha256_before=_sha256(resolved),
                status="planned" if resolved.is_file() else "missing",
            ))
        components_summary.append({
            "component_id": component_id,
            "keep_split": keep_split,
            "representative_keep_path": representative_keep_path,
            "keep_dimensions": {
                "width": keep_width,
                "height": keep_height,
                "pixel_area": keep_area,
            },
            "selection_rule": selection_rule,
            "keep_paths": keep_paths,
            "delete_paths": delete_paths,
            "reason_types": component_reasons,
        })

    summary = {
        "data_dir": str(data_dir),
        "duplicates_csv": str(duplicates_csv),
        "near_duplicate_distance_range_removed": [1, max_near_distance],
        "split_image_counts_before": split_counts,
        "tie_priority": list(tie_priority),
        "n_components": len(components_summary),
        "n_planned_deletions": len(plan),
        "planned_deletions_by_split": dict(Counter(item.split for item in plan)),
        "selection_policy": {
            "primary": "largest pixel area",
            "secondary": "largest short side, then largest long side",
            "tie_break": "smaller split count, then tie_priority",
        },
        "components": components_summary,
    }
    return plan, summary


def apply_deletion_plan(data_dir: Path, plan: Sequence[DeletionItem]) -> List[DeletionItem]:
    """Xoa tep trong plan va cap nhat status tren tung dong bao cao."""
    data_dir = data_dir.resolve()
    processed_at = datetime.now(timezone.utc).isoformat()
    results: List[DeletionItem] = []
    for item in plan:
        result = DeletionItem(**asdict(item))
        result.processed_at_utc = processed_at
        path = _safe_resolve(data_dir, result.path)
        if not path.exists():
            result.status = "missing"
            results.append(result)
            continue
        if not path.is_file():
            result.status = "error"
            result.error = "Target is not a regular file"
            results.append(result)
            continue
        try:
            current_sha = _sha256(path)
            if result.sha256_before and current_sha != result.sha256_before:
                result.status = "error"
                result.error = "File content changed after plan creation"
            else:
                path.unlink()
                result.status = "deleted"
        except OSError as exc:
            result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results


REPORT_FIELDS = tuple(DeletionItem.__dataclass_fields__)


def write_deletion_csv(path: Path, rows: Sequence[DeletionItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
