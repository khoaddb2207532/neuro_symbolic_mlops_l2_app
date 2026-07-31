"""Đánh giá chính xác các checkpoint đã lưu, không train lại model.

Persist accuracy/macro metrics full precision và precision/recall/F1/support
từng lớp dưới dạng JSON. Đây là nguồn metric canonical cho báo cáo.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.dataset import NeuroSymbolicDataset, create_dataloaders
from src.models.cnn import ImageClassificationBaseline
from src.utils.checkpoint import load_model_weights
from src.utils.config import load_params, selected_baseline_architecture
from src.utils.seed import set_seed


def _checkpoint_specs(
    output_dir: Path,
    backbone: str,
    include_bayesian: bool,
) -> List[Tuple[str, Path, Path]]:
    specs = [
        (
            "cnn_baseline",
            output_dir / "baseline_comparison" / backbone
            / "baseline_best.pth",
            output_dir / "baseline_comparison" / backbone,
        ),
        (
            "gflownet_db",
            output_dir / "05_rules_model" / "rule_regularized_best.pth",
            output_dir / "05_rules_model",
        ),
        (
            "random",
            output_dir / "05_rules_model_random"
            / "rule_regularized_best.pth",
            output_dir / "05_rules_model_random",
        ),
        (
            "topk_confidence",
            output_dir / "05_rules_model_topk_confidence"
            / "rule_regularized_best.pth",
            output_dir / "05_rules_model_topk_confidence",
        ),
        (
            "greedy_coverage",
            output_dir / "05_rules_model_greedy_coverage"
            / "rule_regularized_best.pth",
            output_dir / "05_rules_model_greedy_coverage",
        ),
    ]
    if include_bayesian:
        specs.append(
            (
                "gflownet_db_bayesian",
                output_dir / "05b_rules_model_bayesian"
                / "rule_regularized_best.pth",
                output_dir / "05b_rules_model_bayesian",
            )
        )
    return specs


def _class_names(data_dir: str) -> List[str]:
    dataset = NeuroSymbolicDataset(data_dir, "test")
    return [
        name
        for name, _ in sorted(
            dataset.class_to_idx.items(),
            key=lambda item: item[1],
        )
    ]


@torch.no_grad()
def _predict(model, test_loader, device) -> Tuple[np.ndarray, np.ndarray]:
    predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    model.eval()
    for images, batch_labels in test_loader:
        logits, _ = model(images.to(device))
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(labels), np.concatenate(predictions)


def _metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
) -> Dict:
    label_indices = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_indices,
        average=None,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_indices,
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_indices,
        average="weighted",
        zero_division=0,
    )
    per_class = {
        class_name: {
            "class_index": index,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    }
    return {
        "n_test_samples": int(labels.shape[0]),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=label_indices,
        ).astype(int).tolist(),
    }


def _write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: str, include_bayesian: bool) -> Tuple[Path, Path]:
    params = load_params(config_path)
    set_seed(int(params["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(params["output_dir"])
    backbone = selected_baseline_architecture(params)
    class_names = _class_names(params["data_dir"])
    _, _, _, test_loader = create_dataloaders(
        params["data_dir"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        seed=params["seed"],
    )

    specs = _checkpoint_specs(output_dir, backbone, include_bayesian)
    missing = [checkpoint for _, checkpoint, _ in specs if not checkpoint.exists()]
    if missing:
        raise FileNotFoundError(
            "Không thể tính metric chính xác vì thiếu checkpoint:\n- "
            + "\n- ".join(map(str, missing))
        )

    detailed_results = []
    summary_rows = []
    for method, checkpoint, method_dir in specs:
        model = ImageClassificationBaseline(
            architecture=backbone,
            num_classes=params["num_classes"],
            pretrained=False,
        )
        load_model_weights(model, str(checkpoint), device, required=True)
        model = model.to(device)
        labels, predictions = _predict(model, test_loader, device)
        metrics = _metrics(labels, predictions, class_names)
        result = {
            "seed": int(params["seed"]),
            "backbone": backbone,
            "method": method,
            "checkpoint": str(checkpoint),
            **metrics,
        }
        detailed_results.append(result)
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "exact_test_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_rows.append(
            {
                "seed": int(params["seed"]),
                "backbone": backbone,
                "method": method,
                "n_test_samples": metrics["n_test_samples"],
                "test_accuracy": metrics["accuracy"],
                "test_macro_precision": metrics["macro_precision"],
                "test_macro_recall": metrics["macro_recall"],
                "test_f1_macro": metrics["macro_f1"],
                "test_weighted_f1": metrics["weighted_f1"],
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json_path = output_dir / "exact_test_metrics_all_methods.json"
    csv_path = output_dir / "exact_test_metrics_summary.csv"
    json_path.write_text(
        json.dumps(detailed_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(csv_path, summary_rows)

    print("\nEXACT TEST METRICS TỪ CHECKPOINT")
    print(
        f"{'Method':<24} {'Accuracy':>12} {'Macro-P':>12} "
        f"{'Macro-R':>12} {'Macro-F1':>12}"
    )
    for row in summary_rows:
        print(
            f"{row['method']:<24} "
            f"{row['test_accuracy']:>12.8f} "
            f"{row['test_macro_precision']:>12.8f} "
            f"{row['test_macro_recall']:>12.8f} "
            f"{row['test_f1_macro']:>12.8f}"
        )
    print(" -", csv_path)
    print(" -", json_path)
    return csv_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--include-bayesian", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(arguments.config, arguments.include_bayesian)
