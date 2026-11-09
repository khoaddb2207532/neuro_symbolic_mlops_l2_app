"""Vòng lặp huấn luyện chuyên cho Concept Bottleneck Model (CBM).

CBM training khác standard training ở chỗ:
  1. DataLoader trả về (image, label, concept_target) thay vì (image, label).
  2. Loss = task_loss + concept_weight * concept_loss.
  3. Forward trả về (logits, concept_probs, raw_features).

Tách riêng module này (không sửa trainer.py gốc) để:
  - Không phá vỡ các pipeline stage 1-5 hiện có.
  - CBM có thể chạy song song/sau pipeline gốc.
"""
import copy
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from dvclive import Live
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping
from src.training.optimizer import build_optimizer
from src.training.trainer import (
    MONITOR_MODES,
    DEFAULT_CONFIG,
    _NoOpLive,
    save_checkpoint,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def train_one_epoch_cbm(
    model: nn.Module,
    loader: DataLoader,
    task_criterion: nn.Module,
    concept_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    concept_weight: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Một epoch huấn luyện CBM.

    Args:
        model: ConceptBottleneckModel.
        loader: DataLoader trả về (images, labels, concept_targets).
        task_criterion: CrossEntropyLoss cho phân loại label.
        concept_criterion: BCELoss cho concept predictions.
        optimizer: Optimizer.
        device: Device.
        concept_weight: Trọng số concept loss (λ_concept).

    Returns:
        (total_loss, task_loss, concept_loss, accuracy)
    """
    model.train()
    total_loss = total_task = total_concept = 0.0
    total_correct = total_samples = 0

    for batch in loader:
        images, labels, concept_targets = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        concept_targets = concept_targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits, concept_probs, _ = model(images)

        task_loss = task_criterion(logits, labels)
        concept_loss = concept_criterion(concept_probs, concept_targets)

        loss = task_loss + concept_weight * concept_loss
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_task += task_loss.item() * bs
        total_concept += concept_loss.item() * bs
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_samples += bs

    return (
        total_loss / total_samples,
        total_task / total_samples,
        total_concept / total_samples,
        total_correct / total_samples,
    )


def validate_cbm(
    model: nn.Module,
    loader: DataLoader,
    task_criterion: nn.Module,
    concept_criterion: nn.Module,
    device: torch.device,
    concept_weight: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Validate CBM trên tập val.

    Returns:
        (accuracy, total_loss, task_loss, concept_loss)
    """
    model.eval()
    correct = total = 0
    total_loss = total_task = total_concept = 0.0

    with torch.no_grad():
        for batch in loader:
            images, labels, concept_targets = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            concept_targets = concept_targets.to(device, non_blocking=True)

            logits, concept_probs, _ = model(images)
            task_loss = task_criterion(logits, labels)
            concept_loss = concept_criterion(concept_probs, concept_targets)
            loss = task_loss + concept_weight * concept_loss

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_task += task_loss.item() * bs
            total_concept += concept_loss.item() * bs
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += bs

    return (
        correct / total,
        total_loss / total,
        total_task / total,
        total_concept / total,
    )


def train_cbm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cuda",
    concept_weight: float = 1.0,
    smoothing: float = 0.1,
    num_classes: int = 12,
    min_epochs_before_early_stop: int = 0,
    config: Optional[Dict] = None,
) -> Tuple[nn.Module, dict]:
    """Huấn luyện Concept Bottleneck Model.

    Args:
        model: ConceptBottleneckModel.
        train_loader: DataLoader trả về (images, labels, concept_targets).
        val_loader: DataLoader trả về (images, labels, concept_targets).
        concept_weight: λ — trọng số concept loss.
            - concept_weight = 0: chỉ train task loss (CBM trở thành standard model).
            - concept_weight > 0: concept loss ép concept_probs khớp annotations.
            Giá trị khuyến nghị: 0.5 – 5.0 (tuỳ bộ dữ liệu).

    Returns:
        (model_with_best_weights, history_dict)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    cfg.setdefault("lr", lr)
    os.makedirs(cfg["save_dir"], exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    task_criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    concept_criterion = nn.BCELoss()
    optimizer = build_optimizer(model, cfg)

    monitor_metric = cfg.get("monitor_metric", "val_acc")
    if monitor_metric not in MONITOR_MODES:
        raise ValueError(
            f"monitor_metric '{monitor_metric}' không hợp lệ. "
            f"Các lựa chọn: {list(MONITOR_MODES)}"
        )

    stopper = EarlyStopping(
        patience=patience, verbose=True, mode=MONITOR_MODES[monitor_metric]
    )
    ckpt_path = os.path.join(cfg["save_dir"], "cbm_best.pth")

    history = {
        "train_loss": [], "train_task_loss": [], "train_concept_loss": [],
        "train_acc": [], "val_loss": [], "val_task_loss": [],
        "val_concept_loss": [], "val_acc": [],
    }
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    disable_dvclive = os.environ.get("DISABLE_DVCLIVE", "").lower() in {
        "1", "true", "yes",
    }
    live_context = _NoOpLive() if disable_dvclive else Live(
        dir=cfg["dvclive_path"], save_dvc_exp=False,
    )

    start_time = time.time()

    with live_context as live:
        live.log_params({
            "num_epochs": num_epochs, "lr": lr, "patience": patience,
            "concept_weight": concept_weight, "smoothing": smoothing,
            "num_classes": num_classes,
            "n_concepts": getattr(model, "n_concepts", "?"),
            "batch_size": train_loader.batch_size,
            "total_trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        })

        for epoch in range(num_epochs):
            train_loss, train_task, train_concept, train_acc = train_one_epoch_cbm(
                model, train_loader, task_criterion, concept_criterion,
                optimizer, device, concept_weight,
            )
            val_acc, val_loss, val_task, val_concept = validate_cbm(
                model, val_loader, task_criterion, concept_criterion,
                device, concept_weight,
            )

            history["train_loss"].append(train_loss)
            history["train_task_loss"].append(train_task)
            history["train_concept_loss"].append(train_concept)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_task_loss"].append(val_task)
            history["val_concept_loss"].append(val_concept)
            history["val_acc"].append(val_acc)

            if (epoch + 1) % 5 == 0:
                logger.info(
                    "Epoch %d/%d | Loss %.4f (task=%.4f + λ·concept=%.4f) | "
                    "Train Acc %.4f | Val Loss %.4f | Val Acc %.4f",
                    epoch + 1, num_epochs, train_loss, train_task,
                    concept_weight * train_concept, train_acc, val_loss, val_acc,
                )

            live.log_metric("train/loss", train_loss)
            live.log_metric("train/task_loss", train_task)
            live.log_metric("train/concept_loss", train_concept)
            live.log_metric("train/acc", train_acc)
            live.log_metric("val/loss", val_loss)
            live.log_metric("val/task_loss", val_task)
            live.log_metric("val/concept_loss", val_concept)
            live.log_metric("val/acc", val_acc)

            monitor_value = {"val_acc": val_acc, "val_loss": val_loss}[monitor_metric]
            is_best = stopper(monitor_value)

            early_stop_enabled = epoch + 1 >= min_epochs_before_early_stop
            if not early_stop_enabled:
                stopper.counter = 0
                stopper.early_stop = False

            if is_best:
                best_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())
                save_checkpoint(
                    ckpt_path, model, optimizer, epoch + 1, best_acc,
                    class_names=getattr(
                        getattr(train_loader.dataset, "base_dataset", train_loader.dataset),
                        "classes", None,
                    ),
                )
                live.log_metric("best_val_acc", best_acc)

            if early_stop_enabled and stopper.early_stop:
                logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                live.log_metric("early_stop", 1)
                live.next_step()
                break

            live.next_step()

    elapsed = time.time() - start_time
    logger.info(
        "CBM training complete in %.2f minutes | Best Val Acc: %.4f",
        elapsed / 60, best_acc,
    )

    model.load_state_dict(best_weights)
    final_path = os.path.join(cfg["save_dir"], "final_model_weights.pth")
    torch.save(model.state_dict(), final_path)
    return model, history
