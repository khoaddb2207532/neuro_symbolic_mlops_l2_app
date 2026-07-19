"""Dataset protocol gates required before baseline training and RF tuning."""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence


def validate_audit_summary(summary: Mapping, *, fail_on_near_duplicate_leakage: bool = True) -> None:
    """Reject data defects that would make baseline metrics untrustworthy."""
    problems = []
    if summary.get("n_corrupt_images", 0):
        problems.append(f"{summary['n_corrupt_images']} corrupt images")
    if summary.get("class_mismatch"):
        problems.append(f"class mismatch: {summary['class_mismatch']}")
    duplicates = summary.get("duplicates", {})
    if duplicates.get("n_exact_cross_split_groups", 0):
        problems.append(f"{duplicates['n_exact_cross_split_groups']} exact duplicate groups cross splits")
    if fail_on_near_duplicate_leakage and duplicates.get("n_near_cross_split_pairs", 0):
        problems.append(f"{duplicates['n_near_cross_split_pairs']} near-duplicate pairs cross splits")
    if problems:
        raise ValueError("dataset audit gate failed: " + "; ".join(problems))


def validate_split_protocol(split_sizes: Mapping[str, int], val_labels: Sequence[int], *,
                            min_val_fraction: float, min_val_samples_per_class: int,
                            expected_classes: Sequence[int] | None = None) -> dict:
    missing = [name for name in ("train", "val", "test") if split_sizes.get(name, 0) <= 0]
    if missing:
        raise ValueError(f"missing or empty dataset splits: {missing}")
    total = sum(split_sizes[name] for name in ("train", "val", "test"))
    val_fraction = split_sizes["val"] / total
    counts = Counter(int(label) for label in val_labels)
    class_ids = expected_classes if expected_classes is not None else counts.keys()
    underrepresented = {str(label): counts.get(int(label), 0) for label in class_ids
                        if counts.get(int(label), 0) < min_val_samples_per_class}
    if val_fraction < min_val_fraction:
        raise ValueError(f"validation split is too small ({val_fraction:.2%}); required >= {min_val_fraction:.2%}")
    if underrepresented:
        raise ValueError(f"validation has too few samples per class: {underrepresented}; required >= {min_val_samples_per_class}")
    return {"split_sizes": dict(split_sizes), "validation_fraction": val_fraction,
            "validation_class_counts": {str(k): v for k, v in sorted(counts.items())}}
