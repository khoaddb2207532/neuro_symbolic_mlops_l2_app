"""Vòng lặp huấn luyện chuyên cho Concept Bottleneck Model (CBM).

CBM training khác standard training ở chỗ:
  1. DataLoader trả về (image, label, concept_target) thay vì (image, label).
     ``concept_target`` có thể chứa NaN = ignore (xem ``src/data/concepts.py``).
  2. Loss = task_loss + concept_weight * concept_loss, trong đó concept_loss là
     BCE có logits, có ``pos_weight`` theo concept và bỏ qua vị trí ignore.
  3. Forward trả về (logits, concept_probs, raw_features).
  4. Tuỳ chọn huấn luyện 2 giai đoạn: giai đoạn 1 đóng băng backbone (chỉ học
     concept layer + label predictor), giai đoạn 2 mở backbone với learning rate thấp.

Tách riêng module này (không sửa trainer.py gốc) để:
  - Không phá vỡ các pipeline stage 1-5 hiện có.
  - CBM có thể chạy song song/sau pipeline gốc.
"""
import copy
import os
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping
from src.training.concept_loss import masked_concept_bce_with_logits
from src.training.optimizer import build_optimizer
from src.training.trainer import (
    MONITOR_MODES,
    DEFAULT_CONFIG,
    _NoOpMetrics,
    save_checkpoint,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

ConceptCriterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def make_concept_criterion(
    pos_weight: Optional[np.ndarray],
    device: torch.device,
) -> ConceptCriterion:
    """Tạo concept loss nhận (concept_logits, targets) với pos_weight tuỳ chọn."""
    weight = (
        None
        if pos_weight is None
        else torch.as_tensor(np.asarray(pos_weight, dtype=np.float32), device=device)
    )

    def criterion(concept_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return masked_concept_bce_with_logits(concept_logits, targets, pos_weight=weight)

    return criterion


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.backbone.parameters():  # type: ignore[attr-defined]
        parameter.requires_grad_(trainable)


def train_one_epoch_cbm(
    model: nn.Module,
    loader: DataLoader,
    task_criterion: nn.Module,
    concept_criterion: ConceptCriterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    concept_weight: float = 1.0,
    backbone_eval: bool = False,
) -> Tuple[float, float, float, float]:
    """Một epoch huấn luyện CBM.

    Args:
        model: ConceptBottleneckModel.
        loader: DataLoader trả về (images, labels, concept_targets).
        task_criterion: CrossEntropyLoss cho phân loại label.
        concept_criterion: concept loss nhận (concept_logits, targets), xem ``make_concept_criterion``.
        optimizer: Optimizer.
        device: Device.
        concept_weight: Trọng số concept loss (λ_concept).
        backbone_eval: True thì backbone chạy ở chế độ eval (BN không cập nhật thống kê),
            dùng trong giai đoạn 1 khi backbone đóng băng.

    Returns:
        (total_loss, task_loss, concept_loss, accuracy)
    """
    model.train()
    if backbone_eval:
        model.backbone.eval()  # type: ignore[attr-defined]
    total_loss = total_task = total_concept = 0.0
    total_correct = total_samples = 0

    for batch in loader:
        images, labels, concept_targets = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        concept_targets = concept_targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits, _, _, concept_logits = model.forward_with_concept_logits(images)  # type: ignore[attr-defined]

        task_loss = task_criterion(logits, labels)
        concept_loss = concept_criterion(concept_logits, concept_targets)

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
    concept_criterion: ConceptCriterion,
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

            logits, _, _, concept_logits = model.forward_with_concept_logits(images)  # type: ignore[attr-defined]
            task_loss = task_criterion(logits, labels)
            concept_loss = concept_criterion(concept_logits, concept_targets)
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
    concept_pos_weight: Optional[np.ndarray] = None,
    two_stage: bool = False,
    stage1_epochs: int = 0,
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
        concept_pos_weight: pos_weight theo concept, shape [n_concepts]; None = không dùng.
            Tính bằng ``src.training.concept_loss.compute_pos_weight`` trên tập train.
        two_stage: True thì giai đoạn 1 (``stage1_epochs`` epoch đầu) đóng băng backbone,
            giai đoạn 2 mở backbone. Early stopping chỉ bật ở giai đoạn 2.
            Nếu backbone đã đóng băng từ đầu (``freeze_backbone``), không có giai đoạn 2
            nên chế độ này tự tắt kèm cảnh báo.
        stage1_epochs: Số epoch của giai đoạn 1 (chỉ dùng khi ``two_stage``).

    Returns:
        (model_with_best_weights, history_dict)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    cfg.setdefault("lr", lr)
    os.makedirs(cfg["save_dir"], exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if two_stage and not (0 < stage1_epochs < num_epochs):
        raise ValueError(
            f"two_stage cần 0 < stage1_epochs < num_epochs, nhận stage1_epochs={stage1_epochs}, "
            f"num_epochs={num_epochs}."
        )
    backbone_trainable_at_start = any(
        p.requires_grad for p in model.backbone.parameters()  # type: ignore[attr-defined]
    )
    two_stage_active = two_stage and backbone_trainable_at_start
    if two_stage and not backbone_trainable_at_start:
        logger.warning(
            "Backbone đã đóng băng từ đầu (freeze_backbone), không có giai đoạn 2; "
            "tắt two_stage và huấn luyện một giai đoạn."
        )

    task_criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    concept_criterion = make_concept_criterion(concept_pos_weight, device)

    if two_stage_active:
        # build_optimizer chỉ đưa vào các tham số có requires_grad lúc gọi, nên backbone
        # phải trainable khi tạo optimizer để giai đoạn 2 có thể cập nhật nó.
        _set_backbone_trainable(model, True)
    optimizer = build_optimizer(model, cfg)
    if two_stage_active:
        _set_backbone_trainable(model, False)
        logger.info(
            "Two-stage: giai đoạn 1 (epoch 1-%d) đóng băng backbone; giai đoạn 2 mở backbone.",
            stage1_epochs,
        )

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

    live_context = _NoOpMetrics()

    start_time = time.time()

    with live_context as live:
        live.log_params({
            "num_epochs": num_epochs, "lr": lr, "patience": patience,
            "concept_weight": concept_weight, "smoothing": smoothing,
            "num_classes": num_classes,
            "n_concepts": getattr(model, "n_concepts", "?"),
            "batch_size": train_loader.batch_size,
            "concept_pos_weight": concept_pos_weight is not None,
            "two_stage": two_stage_active, "stage1_epochs": stage1_epochs if two_stage_active else 0,
            "total_trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        })

        for epoch in range(num_epochs):
            in_stage1 = two_stage_active and epoch < stage1_epochs
            if two_stage_active and epoch == stage1_epochs:
                _set_backbone_trainable(model, True)
                logger.info("Two-stage: bắt đầu giai đoạn 2 (mở backbone) tại epoch %d.", epoch + 1)

            train_loss, train_task, train_concept, train_acc = train_one_epoch_cbm(
                model, train_loader, task_criterion, concept_criterion,
                optimizer, device, concept_weight, backbone_eval=in_stage1,
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

            # Giai đoạn 1 không early stop: backbone đang đóng băng, val chưa phản ánh
            # khả năng thật của mô hình sau khi fine-tune.
            early_stop_enabled = epoch + 1 >= min_epochs_before_early_stop and not in_stage1
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
