"""Vòng lặp huấn luyện CNN — hỗ trợ:
  * rule-regularized loss (penalty theo luật được GFlowNet chọn) — dùng DUY
    NHẤT `VectorizedRulePenalty` (xem src/rules/penalty.py để biết lý do đã
    gộp bỏ implementation loop trùng lặp trước đây).
  * DVCLive experiment tracking
  * PROGRESSIVE UNFREEZING: chuyển freeze_stage của model theo epoch, theo
    `config["freeze_schedule"]`, đúng chiến lược transfer learning đã chọn
    (xem src/models/cnn.py). Optimizer được build lại mỗi khi stage đổi vì
    param groups (và param nào requires_grad) thay đổi.
"""
import copy
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from dvclive import Live
from torch.utils.data import DataLoader

from src.rules.penalty import VectorizedRulePenalty
from src.rules.rule_types import RuleSet
from src.training.callbacks import EarlyStopping
from src.training.optimizer import build_optimizer, build_scheduler
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Đăng ký "càng cao càng tốt" (max) hay "càng thấp càng tốt" (min) cho từng
# metric có thể theo dõi — thêm metric mới (vd "val_f1") chỉ cần thêm 1 dòng
# ở đây, không phải sửa logic so sánh trong EarlyStopping hay vòng lặp train.
MONITOR_MODES = {
    "val_acc": "max",
    "val_loss": "min",
}

DEFAULT_CONFIG = {
    "weight_decay": 1e-4,
    "use_scheduler": True,
    "scheduler_factor": 0.1,
    "scheduler_patience": 5,
    "scheduler_threshold": 1e-4,
    "scheduler_cooldown": 0,
    "scheduler_min_lr": 1e-6,
    "freeze_bn": True,
    "dvclive_path": "dvclive",
    "save_dir": "outputs",
    "monitor_metric": "val_acc",  # đổi thành "val_loss" nếu muốn early-stop/lưu checkpoint theo loss
    # Chiến lược progressive unfreezing mặc định: epoch -> freeze_stage
    "freeze_schedule": {0: "head_only"},
}


def save_checkpoint(path: str, model: nn.Module, optimizer, epoch: int, best_acc: float, class_names: list):
    """Nơi DUY NHẤT ghi checkpoint model trong toàn bộ pipeline (xem
    EarlyStopping trong callbacks.py — chỉ theo dõi, không ghi file)."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "class_names": class_names,
        },
        path,
    )


def train_one_epoch(model, loader, criterion, optimizer, device, penalty_module=None, freeze_bn=True):
    """penalty_module: nếu không None (VectorizedRulePenalty), được cộng vào
    loss chính. Không còn nhánh loop-based riêng — chỉ một công thức.

    Trả về (train_loss, train_ce, train_penalty, train_acc) — train_acc được
    tính trên chính batch train (không phải eval mode) để theo dõi cùng lúc
    với val_acc trên DVCLive, giúp phát hiện overfit/underfit qua khoảng cách
    train_acc vs val_acc.
    """
    model.train()
    if freeze_bn and hasattr(model, "freeze_bn"):
        model.freeze_bn()

    total_loss = total_ce = total_penalty = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        outputs = model(images)
        logits, features = outputs if isinstance(outputs, (tuple, list)) and len(outputs) == 2 else (outputs, None)

        ce_loss = criterion(logits, labels)
        penalty = penalty_module(features, logits) if penalty_module is not None else torch.tensor(0.0, device=device)

        loss = ce_loss + penalty
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_ce += ce_loss.item() * bs
        total_penalty += penalty.item() * bs
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_samples += bs

    return (
        total_loss / total_samples,
        total_ce / total_samples,
        total_penalty / total_samples,
        total_correct / total_samples,
    )


def validate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total, total_loss / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    rule_set: Optional[RuleSet] = None,
    num_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    device="cuda",
    penalty_weight: float = 0.0,
    use_confidence: bool = True,
    smoothing: float = 0.1,
    num_classes: int = 12,
    config: Optional[Dict] = None,
) -> Tuple[nn.Module, dict]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    cfg.setdefault("lr", lr)
    os.makedirs(cfg["save_dir"], exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    freeze_schedule = cfg["freeze_schedule"]

    def rebuild_optimizer():
        opt = build_optimizer(model, cfg)
        sch = build_scheduler(opt, cfg)
        return opt, sch

    optimizer, scheduler = rebuild_optimizer()

    ckpt_name = os.path.join(cfg["save_dir"], "rule_regularized_best.pth" if penalty_weight > 0 else "baseline_best.pth")
    monitor_metric = cfg["monitor_metric"]
    if monitor_metric not in MONITOR_MODES:
        raise ValueError(
            f"monitor_metric '{monitor_metric}' chưa được đăng ký trong MONITOR_MODES. "
            f"Các lựa chọn hợp lệ: {list(MONITOR_MODES)}."
        )
    stopper = EarlyStopping(patience=patience, verbose=True, mode=MONITOR_MODES[monitor_metric])
    logger.info("EarlyStopping theo dõi '%s' (mode='%s')", monitor_metric, MONITOR_MODES[monitor_metric])

    penalty_module = None
    if penalty_weight > 0 and rule_set is not None:
        penalty_module = VectorizedRulePenalty(
            rule_set, penalty_weight=penalty_weight, use_confidence=use_confidence,
            smoothing=smoothing, num_classes=num_classes,
        )

    history = {
        "train_loss": [], "train_ce": [], "train_penalty": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    start_time = time.time()

    with Live(dir=cfg["dvclive_path"]) as live:
        live.log_params(
            {
                "num_epochs": num_epochs, "lr": lr, "patience": patience,
                "penalty_weight": penalty_weight, "smoothing": smoothing, "num_classes": num_classes,
                "freeze_schedule": str(freeze_schedule),
                "batch_size": train_loader.batch_size,
                "total_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
        )

        for epoch in range(num_epochs):
            # ---- Progressive unfreezing: chuyển stage nếu epoch nằm trong schedule ----
            if epoch in freeze_schedule and hasattr(model, "set_freeze_stage"):
                stage = freeze_schedule[epoch]
                logger.info("Epoch %d: chuyển freeze_stage -> '%s'", epoch, stage)
                model.set_freeze_stage(stage)
                optimizer, scheduler = rebuild_optimizer()

            train_loss, train_ce, train_penalty, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                penalty_module, freeze_bn=cfg.get("freeze_bn", True),
            )
            val_acc, val_loss = validate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["train_ce"].append(train_ce)
            history["train_penalty"].append(train_penalty)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)

            if scheduler is not None:
                scheduler.step(val_loss)

            logger.info(
                "Epoch %d/%d | Train Loss %.4f | Train Acc %.4f | CE %.4f | Penalty %.4f | Val Loss %.4f | Val Acc %.4f | LR %.2e",
                epoch + 1, num_epochs, train_loss, train_acc, train_ce, train_penalty, val_loss, val_acc, current_lr,
            )
            # DVCLive theo dõi đầy đủ 4 metric chính: train_loss, train_acc,
            # val_loss, val_acc — cộng thêm breakdown loss (ce/penalty) và lr.
            live.log_metric("train/loss", train_loss)
            live.log_metric("train/acc", train_acc)
            live.log_metric("train/ce", train_ce)
            live.log_metric("train/penalty", train_penalty)
            live.log_metric("val/loss", val_loss)
            live.log_metric("val/acc", val_acc)
            live.log_metric("lr", current_lr)

            # Giá trị dùng để quyết định "tốt nhất" phụ thuộc monitor_metric
            # (val_acc -> mode='max', val_loss -> mode='min'), xem MONITOR_MODES.
            monitor_value = {"val_acc": val_acc, "val_loss": val_loss}[monitor_metric]

            # Một lời gọi duy nhất quyết định "đây có phải điểm tốt nhất chưa?"
            # -- vừa dùng cho quyết định lưu checkpoint, vừa cho patience counter.
            is_best = stopper(monitor_value)
            if is_best:
                best_acc = val_acc  # vẫn ghi nhận accuracy để log/báo cáo, dù đang theo dõi metric khác
                best_weights = copy.deepcopy(model.state_dict())
                save_checkpoint(
                    ckpt_name, model, optimizer, epoch + 1, best_acc,
                    class_names=getattr(train_loader.dataset, "classes", None),
                )
                live.log_metric("best_val_acc", best_acc)
                live.log_metric(f"best_{monitor_metric}", monitor_value)

            if stopper.early_stop:
                logger.info("Early stopping triggered.")
                live.log_metric("early_stop", 1)
                break

            live.next_step()

    elapsed = time.time() - start_time
    logger.info("Training complete in %.2f minutes | Best Val Acc: %.4f", elapsed / 60, best_acc)

    model.load_state_dict(best_weights)
    final_weight_path = os.path.join(cfg["save_dir"], "final_model_weights.pth")
    torch.save(model.state_dict(), final_weight_path)
    return model, history
