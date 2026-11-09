"""Regenerate report figures and result tables for the six reported methods."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"
SUMMARY = REPORT / "generated" / "summary-5x5x2.csv"
RUN_RESULTS = REPORT / "generated" / "all-results-5x5x2.csv"

METHODS = [
    ("cnn_baseline", "Mô hình nền", "#596275"),
    ("random", "Random", "#8c9bab"),
    ("topk_confidence", "Top-confidence", "#4c78a8"),
    ("greedy_coverage", "Greedy-coverage", "#54a24b"),
    ("gflownet_db", "GFlowNet (Static)", "#b279a2"),
    ("gflownet_db_bayesian", "GFlowNet-Marginal", "#f58518"),
]
BACKBONES = ["alexnet", "mobilenetv3_small", "resnet50", "swin_t", "vit_b_32"]
BACKBONE_LABELS = ["AlexNet", "MobileNetV3-Small", "ResNet-50", "Swin-T", "ViT-B/32"]


def read_summary() -> dict[tuple[str, str, str], dict[str, float]]:
    rows: dict[tuple[str, str, str], dict[str, float]] = {}
    with SUMMARY.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["dataset_id"], row["backbone"], row["method"])
            rows[key] = {
                "acc": 100 * float(row["test_accuracy_mean"]),
                "acc_std": 100 * float(row["test_accuracy_std"]),
                "f1": 100 * float(row["test_f1_macro_mean"]),
                "f1_std": 100 * float(row["test_f1_macro_std"]),
            }
    return rows


def plot_test_performance() -> None:
    data = read_summary()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True)
    x = list(range(len(BACKBONES)))
    panels = [
        ("culture-a", "acc", "(a) Dataset A -- VN ICH", "Accuracy (%)"),
        ("culture-b", "acc", "(b) Dataset B -- FastFood", "Accuracy (%)"),
        ("culture-a", "f1", "(c) Dataset A -- VN ICH", "Macro-F1 (%)"),
        ("culture-b", "f1", "(d) Dataset B -- FastFood", "Macro-F1 (%)"),
    ]
    for axis, (dataset, metric, title, ylabel) in zip(axes.flat, panels):
        for method, label, color in METHODS:
            means = [data[(dataset, backbone, method)][metric] for backbone in BACKBONES]
            stds = [data[(dataset, backbone, method)][f"{metric}_std"] for backbone in BACKBONES]
            axis.errorbar(x, means, yerr=stds, marker="o", markersize=3.5,
                          linewidth=1.25, capsize=2.2, color=color, label=label)
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks(x, BACKBONE_LABELS, rotation=20, ha="right", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, 1.01), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "test-performance.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "test-performance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def vi_number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", "{,}")


def metric_cell(mean: float, std: float, best: bool) -> str:
    value = f"{vi_number(mean)}\\pm{vi_number(std)}"
    return f"$\\mathbf{{{value}}}$" if best else f"${value}$"


def plot_value(data, dataset: str, backbone: str, method: str, metric: str) -> float:
    return data[(dataset, backbone, method)][metric]


def generate_main_results_table() -> None:
    data = read_summary()
    method_tex = {
        "cnn_baseline": "Mô hình nền",
        "random": "Random",
        "topk_confidence": "Top-confidence",
        "greedy_coverage": "Greedy-coverage",
        "gflownet_db": "\\gfn{} (Static)",
        "gflownet_db_bayesian": "\\gfn-Marginal",
    }
    lines = [
        r"\begin{tabular}{ll rrr rrr}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{\textbf{Dataset A -- VN ICH}} & \multicolumn{3}{c}{\textbf{Dataset B -- FastFood}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Backbone & Phương pháp & Acc. (\%) & F1 (\%) & Luật & Acc. (\%) & F1 (\%) & Luật \\",
        r"\midrule",
    ]
    for backbone_index, (backbone, backbone_label) in enumerate(zip(BACKBONES, BACKBONE_LABELS)):
        best = {
            (dataset, metric): max(plot_value(data, dataset, backbone, method, metric)
                                   for method, _, _ in METHODS)
            for dataset in ("culture-a", "culture-b") for metric in ("acc", "f1")
        }
        for method_index, (method, _, _) in enumerate(METHODS):
            prefix = (f"\\multirow{{{len(METHODS)}}}{{*}}{{{backbone_label}}}" if method_index == 0 else "")
            cells = []
            for dataset in ("culture-a", "culture-b"):
                for metric in ("acc", "f1"):
                    mean = plot_value(data, dataset, backbone, method, metric)
                    std = plot_value(data, dataset, backbone, method, f"{metric}_std")
                    cells.append(metric_cell(mean, std, abs(mean - best[(dataset, metric)]) < 1e-10))
                rules = "--"
                if method in {"random", "topk_confidence", "greedy_coverage", "gflownet_db"}:
                    with SUMMARY.open("r", encoding="utf-8", newline="") as stream:
                        row = next(row for row in csv.DictReader(stream)
                                   if row["dataset_id"] == dataset and row["backbone"] == backbone
                                   and row["method"] == method)
                    rules = vi_number(float(row["n_rules_selected_mean"]), 1)
                cells.append(rules)
            lines.append(f"{prefix} & {method_tex[method]} & " + " & ".join(cells) + r" \\")
        if backbone_index < len(BACKBONES) - 1:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (REPORT / "generated" / "main-results-table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_paired_delta_table() -> None:
    rows = list(csv.DictReader((REPORT / "generated" / "paired-delta-by-backbone.csv").open(
        "r", encoding="utf-8", newline="")))
    method_tex = {
        "random": "Random", "topk_confidence": "Top-confidence",
        "greedy_coverage": "Greedy-coverage", "gflownet_db": "\\gfn{} (Static)",
        "gflownet_db_bayesian": "\\gfn-Marginal",
    }
    lines = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"& \multicolumn{2}{c}{Dataset A -- VN ICH} & \multicolumn{2}{c}{Dataset B -- FastFood} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Phương pháp & $\Delta$ Acc. & $\Delta$ F1 & $\Delta$ Acc. & $\Delta$ F1 \\", r"\midrule",
    ]
    for method in method_tex:
        cells = []
        for dataset in ("culture-a", "culture-b"):
            selected = [row for row in rows if row["dataset_id"] == dataset and row["method"] == method]
            for key in ("dacc", "df1"):
                values = [float(row[key]) for row in selected]
                mean, std = statistics.mean(values), statistics.stdev(values)
                cells.append(f"${mean:+.3f}\\pm{std:.3f}$".replace(".", "{,}"))
        lines.append(f"{method_tex[method]} & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (REPORT / "generated" / "paired-delta-table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_marginal_reference_table() -> None:
    with RUN_RESULTS.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {
        (row["dataset_id"], row["backbone"], row["seed"], row["method"]): row
        for row in rows
    }
    references = [
        ("cnn_baseline", "Mô hình nền"),
        ("gflownet_db", r"\gfn{} (Static)"),
        ("random", "Random"),
        ("topk_confidence", "Top-confidence"),
        ("greedy_coverage", "Greedy-coverage"),
    ]
    marginal_rows = [row for row in rows if row["method"] == "gflownet_db_bayesian"]

    def summarize(dataset: str, reference: str) -> tuple[float, float, str]:
        dacc, df1 = [], []
        for marginal in marginal_rows:
            if marginal["dataset_id"] != dataset:
                continue
            key = (dataset, marginal["backbone"], marginal["seed"], reference)
            compared = indexed[key]
            dacc.append(100 * (float(marginal["test_accuracy"]) - float(compared["test_accuracy"])))
            df1.append(100 * (float(marginal["test_f1_macro"]) - float(compared["test_f1_macro"])))
        tolerance = 1e-12
        wins = sum(value > tolerance for value in dacc)
        ties = sum(abs(value) <= tolerance for value in dacc)
        losses = sum(value < -tolerance for value in dacc)
        return statistics.mean(dacc), statistics.mean(df1), f"{wins}/{ties}/{losses}"

    def signed(value: float) -> str:
        return f"{value:+.3f}".replace(".", "{,}")

    lines = [
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"& \multicolumn{3}{c}{Dataset A -- VN ICH} & \multicolumn{3}{c}{Dataset B -- FastFood} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Phương pháp tham chiếu & $\Delta$ Acc. & $\Delta$ F1 & W/T/L & $\Delta$ Acc. & $\Delta$ F1 & W/T/L \\",
        r"\midrule",
    ]
    for reference, label in references:
        cells = []
        for dataset in ("culture-a", "culture-b"):
            dacc, df1, counts = summarize(dataset, reference)
            cells.extend((f"${signed(dacc)}$", f"${signed(df1)}$", counts))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (REPORT / "generated" / "marginal-reference-table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def add_box(axis, xy, width, height, text, face, edge="#385273", fontsize=9):
    box = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.015,rounding_size=0.02",
                         linewidth=1.2, facecolor=face, edgecolor=edge)
    axis.add_patch(box)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center",
              fontsize=fontsize, fontweight="bold")
    return box


def arrow(axis, start, end):
    axis.annotate("", xy=end, xytext=start,
                  arrowprops=dict(arrowstyle="->", color="#4c6382", lw=1.5))


def plot_pipeline() -> None:
    fig, axis = plt.subplots(figsize=(13, 5.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.03, 0.94, "NEXUS: KHÁM PHÁ KHÔNG GIAN LUẬT CHO PHÂN LỚP ẢNH DI SẢN", fontsize=13,
              fontweight="bold", color="#26415e")

    y, w, h = 0.66, 0.13, 0.13
    positions = [0.03, 0.20, 0.37, 0.54]
    labels = ["Ảnh đầu vào", "CNN nền", "Trích xuất\nđặc trưng", "Random Forest"]
    for x, label in zip(positions, labels):
        add_box(axis, (x, y), w, h, label, "#eaf1f8")
    for left, right in zip(positions, positions[1:]):
        arrow(axis, (left + w, y + h / 2), (right, y + h / 2))

    add_box(axis, (0.72, y), 0.12, h, "Tập luật thô", "#fff3d8", edge="#b37a16")
    add_box(axis, (0.86, y), 0.11, h, "Lọc support\nconfidence", "#fff3d8", edge="#b37a16")
    arrow(axis, (0.67, y + h / 2), (0.72, y + h / 2))
    arrow(axis, (0.84, y + h / 2), (0.86, y + h / 2))

    y2 = 0.29
    add_box(axis, (0.08, y2), 0.16, h, "Dự đoán", "#e5f5ea", edge="#3c8760")
    add_box(axis, (0.31, y2), 0.18, h, "CNN được\nregularize", "#e5f5ea", edge="#3c8760")
    add_box(axis, (0.56, y2), 0.18, h, "Kỳ vọng Monte Carlo\ntrên các tập luật", "#eef0fb")
    add_box(axis, (0.81, y2), 0.16, h, "GFlowNet-DB\nDiverse checkpoint", "#eef0fb")
    arrow(axis, (0.31, y2 + h / 2), (0.24, y2 + h / 2))
    arrow(axis, (0.56, y2 + h / 2), (0.49, y2 + h / 2))
    arrow(axis, (0.81, y2 + h / 2), (0.74, y2 + h / 2))
    arrow(axis, (0.915, y), (0.89, y2 + h))
    axis.annotate("Khởi tạo và fine-tune", xy=(0.40, y2 + h), xytext=(0.265, y - 0.06),
                  arrowprops=dict(arrowstyle="->", linestyle="--", color="#70849d", lw=1.2),
                  fontsize=8, color="#53677f")
    axis.text(0.72, 0.13,
              "GFlowNet được đóng băng; gradient chỉ cập nhật mô hình phân loại ảnh.",
              ha="center", fontsize=9, color="#42556d")
    fig.savefig(FIGURES / "pipeline-neuro-symbolic-report.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot_test_performance()
    plot_pipeline()
    generate_main_results_table()
    generate_paired_delta_table()
    generate_marginal_reference_table()
