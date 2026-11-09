"""Build thesis tables and a representative confusion matrix from model bundles."""

from __future__ import annotations

import json
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT / "output" / "bundles_with_classification_reports"
GENERATED_ROOT = ROOT / "report" / "generated"
FIGURE_ROOT = ROOT / "report" / "figures"
METHOD = "gflownet_db_bayesian"
BASELINE_METHOD = "cnn_baseline"

BUNDLES = {
    "alexnet": BUNDLE_ROOT / "bundle_alexnet_with_classification_reports.tar.gz",
    "mobilenetv3_small": (
        BUNDLE_ROOT / "bundle_mobilenetv3_small_with_classification_reports.tar.gz"
    ),
    "resnet50": BUNDLE_ROOT / "bundle_resnet50_with_classification_reports.tar.gz",
    "swin_t": BUNDLE_ROOT / "bundle_swin_t_with_classification_reports.tar.gz",
    "vit_b_32": BUNDLE_ROOT / "bundle_vit_b_32_with_classification_reports.tar.gz",
}
BACKBONE_NAMES = {
    "alexnet": "AlexNet",
    "mobilenetv3_small": "MobileNetV3-Small",
    "resnet50": "ResNet-50",
    "swin_t": "Swin-T",
    "vit_b_32": "ViT-B/32",
}
DATASET_NAMES = {
    "culture-a": r"Dataset A -- \dsa{}",
    "culture-b": r"Dataset B -- \dsb{}",
}
CLASS_NAMES = {
    "cho-noi-cai-rang": "Chợ nổi Cái Răng",
    "chua": "Chùa",
    "dieu-mua-truyen-thong": "Điệu múa truyền thống",
    "don-ca-tai-tu": "Đờn ca tài tử",
    "dua-bo-bay-nui": "Đua bò Bảy Núi",
    "le-hoi-nghinh-ong": "Lễ hội Nghinh Ông",
    "nghe-dan-tre": "Nghề đan tre",
    "nghe-det-chieu": "Nghề dệt chiếu",
    "nghe-thuat-san-khau-du-ke-khmer": "Nghệ thuật sân khấu Dù Kê Khmer",
    "ok-om-bok": "Ok Om Bok",
    "phong-tuc-cuoi-truyen-thong": "Phong tục cưới truyền thống",
    "via-ba-chua-xu-nui-sam": "Vía Bà Chúa Xứ Núi Sam",
    "Baked Potato": "Baked Potato",
    "Burger": "Burger",
    "Crispy Chicken": "Crispy Chicken",
    "Donut": "Donut",
    "Fries": "Fries",
    "Hot Dog": "Hot Dog",
    "Pizza": "Pizza",
    "Sandwich": "Sandwich",
    "Taco": "Taco",
    "Taquito": "Taquito",
}


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def metric(values: list[float]) -> str:
    value = f"{statistics.mean(values):.3f} $\\pm$ {sample_sd(values):.3f}"
    return value.replace(".", ",")


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def load_results(method: str = METHOD) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for backbone, archive_path in BUNDLES.items():
        with tarfile.open(archive_path) as archive:
            metric_files = sorted(
                name
                for name in archive.getnames()
                if name.endswith("/exact_test_metrics.json")
                and "/runs/" in name
            )
            if len(metric_files) != 10:
                raise ValueError(f"{backbone}: expected 10 exact metric files")
            for name in metric_files:
                run_id = Path(name).parent.name
                dataset_id = run_id.split("__", 1)[0]
                payload = json.load(archive.extractfile(name))
                matches = [row for row in payload if row["method"] == method]
                if len(matches) != 1:
                    raise ValueError(f"{run_id}: expected one {method} result")
                grouped[(backbone, dataset_id)].append(matches[0])
    if any(len(rows) != 5 for rows in grouped.values()) or len(grouped) != 10:
        raise ValueError("Classification results are incomplete")
    return grouped


def main_summary(grouped: dict[tuple[str, str], list[dict]]) -> str:
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Bộ dữ liệu & Backbone & Accuracy & Precision & Recall & F1-score \\",
        r"\midrule",
    ]
    for dataset_id in ("culture-a", "culture-b"):
        for index, backbone in enumerate(BUNDLES):
            rows = grouped[(backbone, dataset_id)]
            dataset_cell = DATASET_NAMES[dataset_id] if index == 0 else ""
            lines.append(
                " & ".join(
                    [
                        dataset_cell,
                        BACKBONE_NAMES[backbone],
                        metric([float(row["accuracy"]) for row in rows]),
                        metric([float(row["macro_precision"]) for row in rows]),
                        metric([float(row["macro_recall"]) for row in rows]),
                        metric([float(row["macro_f1"]) for row in rows]),
                    ]
                )
                + r" \\"
            )
        if dataset_id == "culture-a":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def detail_table(backbone: str, dataset_id: str, rows: list[dict]) -> str:
    ordered_classes = sorted(
        rows[0]["per_class"],
        key=lambda name: int(rows[0]["per_class"][name]["class_index"]),
    )
    total_support = int(rows[0]["n_test_samples"])
    lines = [
        r"\begin{subtable}[t]{\linewidth}",
        r"\centering",
        (
            r"\caption{"
            + DATASET_NAMES[dataset_id]
            + r".}"
        ),
        rf"\label{{tab:classification-{dataset_id}-{backbone}}}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}Xcccc}",
        r"\toprule",
        r"Lớp & Precision & Recall & F1-score & Support \\",
        r"\midrule",
    ]
    for class_name in ordered_classes:
        class_rows = [row["per_class"][class_name] for row in rows]
        supports = {int(row["support"]) for row in class_rows}
        if len(supports) != 1:
            raise ValueError(f"Support changed: {backbone}/{dataset_id}/{class_name}")
        label = latex_escape(CLASS_NAMES.get(class_name, class_name))
        lines.append(
            " & ".join(
                [
                    label,
                    metric([float(row["precision"]) for row in class_rows]),
                    metric([float(row["recall"]) for row in class_rows]),
                    metric([float(row["f1"]) for row in class_rows]),
                    str(supports.pop()),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\midrule",
            "Accuracy & -- & -- & "
            + metric([float(row["accuracy"]) for row in rows])
            + f" & {total_support}"
            + r" \\",
            "Macro avg & "
            + " & ".join(
                [
                    metric([float(row["macro_precision"]) for row in rows]),
                    metric([float(row["macro_recall"]) for row in rows]),
                    metric([float(row["macro_f1"]) for row in rows]),
                    str(total_support),
                ]
            )
            + r" \\",
            "Weighted avg & "
            + " & ".join(
                [
                    metric([float(row["weighted_precision"]) for row in rows]),
                    metric([float(row["weighted_recall"]) for row in rows]),
                    metric([float(row["weighted_f1"]) for row in rows]),
                    str(total_support),
                ]
            )
            + r" \\",
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{subtable}",
        ]
    )
    return "\n".join(lines)


def appendix(grouped: dict[tuple[str, str], list[dict]]) -> str:
    parts = [r"\clearpage"]
    for backbone in BUNDLES:
        parts.extend(
            [
                r"\begin{table}[H]",
                r"\centering",
                (
                    r"\caption{Classification report đầy đủ của \gfn-Marginal với "
                    + BACKBONE_NAMES[backbone]
                    + r". Mỗi chỉ số là trung bình $\pm$ độ lệch chuẩn qua năm seed.}"
                ),
                rf"\label{{tab:classification-full-{backbone}}}",
                r"\captionsetup[subtable]{font=small,skip=4pt}",
                detail_table(backbone, "culture-a", grouped[(backbone, "culture-a")]),
                r"\vspace{1.2em}",
                detail_table(backbone, "culture-b", grouped[(backbone, "culture-b")]),
                r"\end{table}",
                r"\clearpage",
            ]
        )
    return "\n\n".join(parts) + "\n"


def signed_metric(values: list[float]) -> str:
    value = f"{statistics.mean(values):+.3f} $\\pm$ {sample_sd(values):.3f}"
    return value.replace(".", ",")


def baseline_delta_tables(
    baseline: dict[tuple[str, str], list[dict]],
    nexus: dict[tuple[str, str], list[dict]],
) -> str:
    """Build the Dataset A per-class comparison used in the main results chapter."""
    dataset_id = "culture-a"
    summary_rows: list[str] = []
    detail_parts: list[str] = []

    for backbone in BUNDLES:
        baseline_rows = sorted(baseline[(backbone, dataset_id)], key=lambda row: row["seed"])
        nexus_rows = sorted(nexus[(backbone, dataset_id)], key=lambda row: row["seed"])
        if [row["seed"] for row in baseline_rows] != [row["seed"] for row in nexus_rows]:
            raise ValueError(f"Seed mismatch for {backbone}/{dataset_id}")

        ordered_classes = sorted(
            baseline_rows[0]["per_class"],
            key=lambda name: int(baseline_rows[0]["per_class"][name]["class_index"]),
        )
        class_deltas: dict[str, list[float]] = {}
        table_rows: list[str] = []
        for class_name in ordered_classes:
            base_class = [row["per_class"][class_name] for row in baseline_rows]
            nexus_class = [row["per_class"][class_name] for row in nexus_rows]
            delta_p = [float(n["precision"]) - float(b["precision"]) for b, n in zip(base_class, nexus_class)]
            delta_r = [float(n["recall"]) - float(b["recall"]) for b, n in zip(base_class, nexus_class)]
            delta_f1 = [float(n["f1"]) - float(b["f1"]) for b, n in zip(base_class, nexus_class)]
            class_deltas[class_name] = delta_f1
            table_rows.append(
                " & ".join(
                    [
                        latex_escape(CLASS_NAMES.get(class_name, class_name)),
                        metric([float(row["precision"]) for row in base_class]),
                        metric([float(row["recall"]) for row in base_class]),
                        metric([float(row["f1"]) for row in base_class]),
                        signed_metric(delta_p),
                        signed_metric(delta_r),
                        signed_metric(delta_f1),
                    ]
                )
                + r" \\"
            )

        mean_delta = {name: statistics.mean(values) for name, values in class_deltas.items()}
        best = max(mean_delta, key=mean_delta.get)
        worst = min(mean_delta, key=mean_delta.get)
        improved = sum(value > 0 for value in mean_delta.values())
        summary_rows.append(
            " & ".join(
                [
                    BACKBONE_NAMES[backbone],
                    f"{improved}/{len(ordered_classes)}",
                    latex_escape(CLASS_NAMES.get(best, best)),
                    signed_metric(class_deltas[best]),
                    latex_escape(CLASS_NAMES.get(worst, worst)),
                    signed_metric(class_deltas[worst]),
                ]
            )
            + r" \\"
        )

        detail_parts.extend(
            [
                r"\begin{table}[H]",
                r"\centering",
                (
                    r"\caption{Classification report baseline và mức thay đổi của \gfn-Marginal "
                    + "với "
                    + BACKBONE_NAMES[backbone]
                    + r" trên Dataset A. Các cột $\Delta$ được tính theo \gfn-Marginal trừ baseline; "
                    + r"mỗi giá trị là trung bình $\pm$ độ lệch chuẩn qua năm seed.}"
                ),
                rf"\label{{tab:classification-delta-culture-a-{backbone}}}",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{2.2pt}",
                r"\resizebox{\linewidth}{!}{%",
                r"\begin{tabular}{lcccccc}",
                r"\toprule",
                r"Lớp & Baseline P & Baseline R & Baseline F1 & $\Delta P$ & $\Delta R$ & $\Delta F1$ \\ ",
                r"\midrule",
                *table_rows,
                r"\bottomrule",
                r"\end{tabular}%",
                r"}",
                r"\end{table}",
            ]
        )

    summary = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Tóm tắt biến thiên F1-score theo lớp của \gfn-Marginal so với baseline trên Dataset A.}",
        r"\label{tab:classification-delta-summary}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lclclc}",
        r"\toprule",
        r"Backbone & Số lớp tăng & Tăng nhiều nhất & $\Delta F1$ & Giảm nhiều nhất & $\Delta F1$ \\ ",
        r"\midrule",
        *summary_rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(summary + detail_parts) + "\n"


def extract_representative_figure() -> None:
    archive_path = BUNDLES["swin_t"]
    suffix = "/thesis_figures/culture-a_gflownet_marginal_confusion_matrix_normalized.png"
    with tarfile.open(archive_path) as archive:
        matches = [name for name in archive.getnames() if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("Representative confusion matrix is missing")
        image = archive.extractfile(matches[0]).read()
    (FIGURE_ROOT / "classification-confusion-culture-a-swin-t.png").write_bytes(image)


def main() -> None:
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    grouped = load_results()
    baseline_grouped = load_results(BASELINE_METHOD)
    (GENERATED_ROOT / "classification-report-summary.tex").write_text(
        main_summary(grouped), encoding="utf-8"
    )
    (GENERATED_ROOT / "classification-report-appendix.tex").write_text(
        appendix(grouped), encoding="utf-8"
    )
    (GENERATED_ROOT / "classification-baseline-vs-marginal.tex").write_text(
        baseline_delta_tables(baseline_grouped, grouped), encoding="utf-8"
    )
    extract_representative_figure()


if __name__ == "__main__":
    main()
