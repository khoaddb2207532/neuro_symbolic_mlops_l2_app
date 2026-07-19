"""Đánh giá model và trực quan hoá kết quả huấn luyện."""
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_classification_metrics(model, dataloader, device) -> Dict[str, Any]:
    """Return accuracy/macro-F1 for an evaluation split without presentation side effects."""
    model = model.to(device).eval()
    predictions, labels = [], []
    with torch.no_grad():
        for images, targets in dataloader:
            logits, _ = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(targets.tolist())
    if not labels:
        raise ValueError("cannot evaluate an empty split")
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "n_samples": len(labels)}


def evaluate_model_performance(
    model: Optional[nn.Module],
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    title: str = "Model Evaluation",
    output_dir: Optional[str] = None,
) -> float:
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report_str = classification_report(all_labels, all_preds, target_names=class_names)
    logger.info("%s | Overall Accuracy: %.2f%%", title, acc * 100)

    cm = confusion_matrix(all_labels, all_preds)
    fig = plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {title}")
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_title = title.replace(" ", "_").lower()
        with open(os.path.join(output_dir, f"{safe_title}_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"Overall Accuracy: {acc*100:.2f}%\n\n{report_str}")
        fig.savefig(os.path.join(output_dir, f"{safe_title}_confusion_matrix.png"), dpi=300)
    plt.close(fig)
    return acc


def plot_training_history(
    history: Optional[Dict[str, Any]] = None,
    history_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    title_suffix: str = "",
) -> None:
    if history_path is not None and os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    if history is None:
        raise ValueError("Cần cung cấp 'history' hoặc 'history_path' hợp lệ!")

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if len(epochs) == 0:
        logger.warning("Dữ liệu history trống, không thể vẽ biểu đồ.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Total Loss", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], "g-", label="Val Loss", linewidth=2)
    axes[0].set_title(f"Loss {title_suffix}".strip())
    axes[0].set_xlabel("Epochs")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    if "train_acc" in history:
        axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    if "val_acc" in history:
        axes[1].plot(epochs, history["val_acc"], "g-", label="Val Acc", linewidth=2)
    axes[1].set_title(f"Accuracy {title_suffix}".strip())
    axes[1].set_xlabel("Epochs")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_suffix = title_suffix.strip().replace(" ", "_").lower()
        filename = f"training_metrics_{safe_suffix}.png" if safe_suffix else "training_metrics.png"
        fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
