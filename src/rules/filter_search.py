"""Histogram valley detection and elbow selection for faithful RF leaves."""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def fidelity_valley(values: Iterable[float], bins: int = 20) -> float | None:
    counts, edges = np.histogram(np.asarray(list(values), dtype=float), bins=bins, range=(0.0, 1.0))
    smooth = np.convolve(counts, np.array([0.25, 0.5, 0.25]), mode="same")
    peaks = [i for i in range(1, len(smooth) - 1)
             if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]]
    if len(peaks) < 2:
        return None
    first, second = sorted(sorted(peaks, key=lambda i: smooth[i], reverse=True)[:2])
    if second - first < 2:
        return None
    valley = first + int(np.argmin(smooth[first: second + 1]))
    return float((edges[valley] + edges[valley + 1]) / 2)


def threshold_ablation(stats: pd.DataFrame, thresholds, min_coverages) -> List[Dict]:
    total_assignments = max(float(stats.coverage.sum()), 1.0)
    rows = []
    for threshold in thresholds:
        for minimum in min_coverages:
            kept = stats[(stats.wilson_fidelity_lower > threshold) & (stats.coverage > minimum)]
            rows.append({"fidelity_threshold": float(threshold), "min_coverage": int(minimum),
                         "retained_leaves": int(len(kept)),
                         "coverage_ratio": float(kept.coverage.sum() / total_assignments)})
    return rows


def select_filter_elbow(rows: List[Dict], random_floor: float, valley: float | None) -> Dict:
    floor = max(random_floor, valley or 0.0)
    feasible = [row for row in rows if row["fidelity_threshold"] > floor]
    if not feasible:
        feasible = [row for row in rows if row["fidelity_threshold"] > random_floor]
    max_leaves = max((row["retained_leaves"] for row in rows), default=1)
    for row in feasible:
        retained_ratio = row["retained_leaves"] / max(max_leaves, 1)
        # Balanced knee: preserve assignment coverage while removing unreliable leaves.
        row["selection_score"] = row["coverage_ratio"] + (1.0 - retained_ratio)
    return max(feasible, key=lambda row: (row["selection_score"], row["coverage_ratio"]))


def write_filter_search_csv(log_dir: str, rows: List[Dict]) -> str:
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = os.path.join(log_dir, f"search_filter_{stamp}.csv")
    fields = sorted({key for row in rows for key in row})
    with open(path, "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return path
