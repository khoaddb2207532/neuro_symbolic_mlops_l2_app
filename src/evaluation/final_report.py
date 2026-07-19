"""Ablation report generation from existing timestamped logs only."""
from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


LOG_SPECS = {
    "rf": ("search_rf_*.csv", "weighted_fidelity"),
    "filter": ("search_filter_*.csv", "coverage_ratio"),
    "gfn": ("search_gfn_*.csv", "objective"),
    "regularize": ("search_regularize_*.csv", "accuracy"),
    "drift": ("drift_check_*.csv", "fidelity"),
}


def build_sensitivity_plots(log_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    generated = []
    for group, (pattern, metric) in LOG_SPECS.items():
        paths = sorted(glob.glob(os.path.join(log_dir, pattern)))
        if not paths:
            continue
        frames = [pd.read_csv(path) for path in paths]
        data = pd.concat(frames, ignore_index=True)
        if metric not in data or not pd.api.types.is_numeric_dtype(data[metric]):
            continue
        candidates = [column for column in data.columns
                      if column != metric and pd.api.types.is_numeric_dtype(data[column])]
        if not candidates:
            continue
        figure, axes = plt.subplots(len(candidates), 1, figsize=(7, max(4, 3 * len(candidates))))
        axes = [axes] if len(candidates) == 1 else axes
        for axis, column in zip(axes, candidates):
            axis.scatter(data[column], data[metric], alpha=0.55, s=18)
            axis.set(xlabel=column, ylabel=metric, title=f"{group}: {column} sensitivity")
        figure.tight_layout()
        path = os.path.join(output_dir, f"{group}_sensitivity.png")
        figure.savefig(path, dpi=170); plt.close(figure); generated.append(path)
    return generated


def sum_logged_duration(path: str | None) -> float:
    if not path or not os.path.exists(path):
        return 0.0
    data = pd.read_csv(path)
    return float(data["duration_seconds"].fillna(0).sum()) if "duration_seconds" in data else 0.0
