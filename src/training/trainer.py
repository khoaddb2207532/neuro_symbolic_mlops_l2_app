"""Vòng lặp huấn luyện CNN — hỗ trợ:
  * rule-regularized loss (penalty theo luật được GFlowNet chọn) — dùng DUY
    NHẤT `VectorizedRulePenalty` (xem src/rules/penalty.py để biết lý do đã
    gộp bỏ implementation loop trùng lặp trước đây).
  * DVCLive experiment tracking
"""
import copy
import os
import random
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from dvclive import Live
from torch.utils.data import DataLoader

from src.rules.penalty import VectorizedRulePenalty
from src.rules.rule_types import RuleSet
from src.training.callbacks import EarlyStopping
from src.training.optimizer import build_optimizer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class _NoOpLive:
    """DVCLive-compatible no-op used by concurrent dual-GPU workers."""

    def __init__(self):
        self.summary = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def log_params(self, *_args, **_kwargs):
        return None

    def log_metric(self, *_args, **_kwargs):
        return None

    def next_step(self):
        return None

# Đăng ký "càng cao càng tốt" (max) hay "càng thấp càng tốt" (min) cho từng
# metric có thể theo dõi — thêm metric mới (vd "val_f1") chỉ cần thêm 1 dòng
# ở đây, không phải sửa logic so sánh trong EarlyStopping hay vòng lặp train.
MONITOR_MODES = {
    "val_acc": "max",
    "val_loss": "min",
}

DEFAULT_CONFIG = {
    "weight_decay": 1e-4,
    "dvclive_path": "dvclive",
    "save_dir": "outputs",
    "monitor_metric": "val_acc",  # đổi thành "val_loss" nếu muốn early-stop/lưu checkpoint theo loss
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


def _rng_state(train_loader: DataLoader) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    generator = getattr(train_loader, "generator", None)
    state["train_loader_generator"] = (
        generator.get_state() if generator is not None else None
    )
    return state


def _restore_rng_state(state: dict, train_loader: DataLoader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    generator = getattr(train_loader, "generator", None)
    if generator is not None and state.get("train_loader_generator") is not None:
        generator.set_state(state["train_loader_generator"].cpu())


def _save_resume_checkpoint(path: str, state: dict) -> None:
    """Atomically replace the last-epoch checkpoint.

    A crash during ``torch.save`` leaves the previous completed epoch intact.
    """
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def train_one_epoch(
    model, loader, criterion, optimizer, device, penalty_module=None,
):
    """penalty_module: nếu không None (VectorizedRulePenalty hoặc
    BayesianRuleMarginalization — cùng chữ ký forward(features, logits)),
    được cộng vào loss chính. Không còn nhánh loop-based riêng — chỉ một
    công thức.

    Trả về (train_loss, train_ce, train_penalty, train_acc, coverage_stats).
    train_acc được tính trên chính batch train (không phải eval mode) để
    theo dõi cùng lúc với val_acc trên DVCLive, giúp phát hiện overfit/
    underfit qua khoảng cách train_acc vs val_acc.

    coverage_stats (dict rỗng nếu không có penalty_module hoặc module không
    hỗ trợ `last_coverage_stats()`): thống kê xem luật có thực sự "sống"
    (đóng góp gradient) hay không, tổng hợp theo batch-size-weighted average
    qua các batch trong epoch — xem src/rules/penalty.py::last_coverage_stats().
    """
    model.train()

    total_loss = total_ce = total_penalty = 0.0
    total_correct = 0
    total_samples = 0
    # Tích luỹ coverage stats qua các batch (weighted theo batch size, giống
    # cách total_loss/total_ce/... được tích luỹ ở trên).
    sum_coverage_ratio = 0.0
    sum_mean_sat = 0.0
    n_rules_total = 0
    has_coverage = penalty_module is not None and hasattr(penalty_module, "last_coverage_stats")
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        outputs = model(images)
        logits, features = outputs if isinstance(outputs, (tuple, list)) and len(outputs) == 2 else (outputs, None)

        ce_loss = criterion(logits, labels)
        if penalty_module is None:
            penalty = torch.tensor(0.0, device=device)
        elif getattr(penalty_module, "uses_images", False):
            penalty = penalty_module(images, logits)
        else:
            penalty = penalty_module(features, logits)

        loss = ce_loss + penalty
        loss.backward()

        optimizer.step()

        bs = images.size(0)
        loss_value = ce_loss.item() + penalty.item()
        total_loss += loss_value * bs
        total_ce += ce_loss.item() * bs
        total_penalty += penalty.item() * bs
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_samples += bs

        if has_coverage:
            stats = penalty_module.last_coverage_stats()
            n_rules_total = stats["n_rules_total"]  # không đổi giữa các batch
            coverage_ratio = (
                stats["n_rules_active_this_batch"] / n_rules_total if n_rules_total > 0 else 0.0
            )
            sum_coverage_ratio += coverage_ratio * bs
            sum_mean_sat += stats["mean_rule_sat"] * bs

    coverage_stats = {}
    if has_coverage and total_samples > 0:
        coverage_stats = {
            "n_rules_total": n_rules_total,
            "coverage_ratio": sum_coverage_ratio / total_samples,
            "mean_rule_sat": sum_mean_sat / total_samples,
        }
    return (
        total_loss / total_samples,
        total_ce / total_samples,
        total_penalty / total_samples,
        total_correct / total_samples,
        coverage_stats,
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
    initial_temp: float = 2.0,
    final_temp: float = 15.0,
    temp_warmup_epochs: int = 2,
    temp_anneal_epochs: int = 10,
    min_epochs_before_early_stop: int = 0,
    config: Optional[Dict] = None,
    penalty_module: Optional[nn.Module] = None,
) -> Tuple[nn.Module, dict]:
    """penalty_module: nếu được truyền sẵn (vd
    `src.rules.bayesian_penalty.BayesianRuleMarginalization`), dùng TRỰC
    TIẾP module này thay vì tự build `VectorizedRulePenalty` từ `rule_set` —
    cho phép cắm chiến lược rule-regularization khác (Bayesian
    marginalization qua frozen GFlowNet sampler) mà không phải sửa vòng lặp
    train — cắm module khác vào chỉ bằng cách truyền tham số này. Nếu None
    (mặc định, hành vi cũ), vẫn build `VectorizedRulePenalty`
    từ `rule_set`/`penalty_weight` như trước.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    cfg.setdefault("lr", lr)
    os.makedirs(cfg["save_dir"], exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    optimizer = build_optimizer(model, cfg)

    is_rule_regularized_run = penalty_weight > 0 or penalty_module is not None
    if temp_warmup_epochs < 0 or temp_anneal_epochs < 1:
        raise ValueError("temp_warmup_epochs >= 0 và temp_anneal_epochs >= 1.")
    if min_epochs_before_early_stop < 0:
        raise ValueError("min_epochs_before_early_stop không được âm.")
    if is_rule_regularized_run:
        anneal_end_epoch = temp_warmup_epochs + temp_anneal_epochs
        if min_epochs_before_early_stop < anneal_end_epoch:
            logger.warning(
                "min_epochs_before_early_stop=%d nhỏ hơn thời điểm hoàn tất ủ T=%d; "
                "tự nâng lên %d để early stopping không cắt ngang lịch ủ.",
                min_epochs_before_early_stop,
                anneal_end_epoch,
                anneal_end_epoch,
            )
            min_epochs_before_early_stop = anneal_end_epoch
    ckpt_name = os.path.join(cfg["save_dir"], "rule_regularized_best.pth" if is_rule_regularized_run else "baseline_best.pth")
    monitor_metric = cfg["monitor_metric"]
    if monitor_metric not in MONITOR_MODES:
        raise ValueError(
            f"monitor_metric '{monitor_metric}' chưa được đăng ký trong MONITOR_MODES. "
            f"Các lựa chọn hợp lệ: {list(MONITOR_MODES)}."
        )
    stopper = EarlyStopping(patience=patience, verbose=True, mode=MONITOR_MODES[monitor_metric])
    logger.info("EarlyStopping theo dõi '%s' (mode='%s')", monitor_metric, MONITOR_MODES[monitor_metric])

    if penalty_module is None and penalty_weight > 0 and rule_set is not None:
        penalty_module = VectorizedRulePenalty(
            rule_set, penalty_weight=penalty_weight, use_confidence=use_confidence,
            smoothing=smoothing, num_classes=num_classes,
            initial_temp=initial_temp, final_temp=final_temp,
            temp_warmup_epochs=temp_warmup_epochs,
            temp_anneal_epochs=temp_anneal_epochs,
        ).to(device)
    elif penalty_module is not None:
        penalty_module = penalty_module.to(device)
        logger.info("Dùng penalty_module được truyền sẵn: %s", type(penalty_module).__name__)

    history = {
        "train_loss": [], "train_ce": [], "train_penalty": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    start_epoch = 0
    resume_enabled = bool(cfg.get("resume", False))
    resume_path = cfg.get(
        "resume_checkpoint_path",
        os.path.join(cfg["save_dir"], "training_last.pth"),
    )
    resume_identity = cfg.get("resume_identity", {})

    if resume_enabled and os.path.exists(resume_path):
        try:
            resume_state = torch.load(
                resume_path, map_location=device, weights_only=False
            )
        except TypeError:  # PyTorch < 2.6
            resume_state = torch.load(resume_path, map_location=device)
        saved_identity = resume_state.get("resume_identity", {})
        if saved_identity != resume_identity:
            raise ValueError(
                "Resume checkpoint belongs to a different experiment: "
                f"saved={saved_identity}, current={resume_identity}."
            )
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        history = resume_state["history"]
        best_acc = float(resume_state["best_acc"])
        best_weights = resume_state["best_weights"]
        start_epoch = int(resume_state["epoch"])
        stopper_state = resume_state["early_stopping"]
        stopper.counter = int(stopper_state["counter"])
        stopper.best_score = stopper_state["best_score"]
        stopper.early_stop = bool(stopper_state["early_stop"])
        if penalty_module is not None and resume_state.get("penalty_state") is not None:
            if not hasattr(penalty_module, "load_resume_state_dict"):
                raise TypeError(
                    f"{type(penalty_module).__name__} cannot restore penalty state."
                )
            penalty_module.load_resume_state_dict(resume_state["penalty_state"])
        _restore_rng_state(resume_state["rng_state"], train_loader)
        logger.info(
            "Resume training từ epoch %d/%d tại %s.",
            start_epoch,
            num_epochs,
            resume_path,
        )
        if resume_state.get("completed", False):
            model.load_state_dict(best_weights)
            torch.save(
                model.state_dict(),
                os.path.join(cfg["save_dir"], "final_model_weights.pth"),
            )
            logger.info("Checkpoint resume đã hoàn tất; không train lại.")
            return model, history

    def build_resume_state(epoch: int, *, completed: bool) -> dict:
        penalty_state = None
        if penalty_module is not None and hasattr(
            penalty_module, "resume_state_dict"
        ):
            penalty_state = penalty_module.resume_state_dict()
        return {
            "version": 1,
            "epoch": epoch,
            "completed": completed,
            "resume_identity": resume_identity,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "best_weights": best_weights,
            "history": history,
            "early_stopping": {
                "counter": stopper.counter,
                "best_score": stopper.best_score,
                "early_stop": stopper.early_stop,
            },
            "penalty_state": penalty_state,
            "rng_state": _rng_state(train_loader),
        }

    start_time = time.time()
    last_completed_epoch = start_epoch

    # DVC itself orchestrates the pipeline.  Disabling DVCLive's implicit
    # experiment save avoids checking outputs from unrelated stages and avoids
    # repeatedly appending generated params/metrics/plots to dvc.yaml.
    disable_dvclive = os.environ.get("DISABLE_DVCLIVE", "").lower() in {
        "1", "true", "yes",
    }
    live_context = _NoOpLive() if disable_dvclive else Live(
        dir=cfg["dvclive_path"],
        save_dvc_exp=False,
        resume=resume_enabled and start_epoch > 0,
    )
    with live_context as live:
        live.log_params(
            {
                "num_epochs": num_epochs, "lr": lr, "patience": patience,
                "penalty_weight": penalty_weight, "smoothing": smoothing, "num_classes": num_classes,
                "batch_size": train_loader.batch_size,
                "total_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "initial_temp": initial_temp, 
                "final_temp": final_temp,
                "temp_warmup_epochs": temp_warmup_epochs,
                "temp_anneal_epochs": temp_anneal_epochs,
                "min_epochs_before_early_stop": min_epochs_before_early_stop,
            }
        )

        for epoch in range(start_epoch, num_epochs):
            # ---- Ủ nhiệt độ khớp luật: mềm ở epoch đầu -> cứng dần về cuối ----
            if penalty_module is not None:
                penalty_module.update_temperature(epoch)
                live.log_metric("rules/temperature", penalty_module._temperature.item())

            train_loss, train_ce, train_penalty, train_acc, coverage_stats = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                penalty_module,
            )
            val_acc, val_loss = validate(model, val_loader, criterion, device)
            lr_details = " | ".join([
                f"{g.get('name', f'group{i}')}:{g['lr']:.2e}" 
                for i, g in enumerate(optimizer.param_groups)
            ])

            history["train_loss"].append(train_loss)
            history["train_ce"].append(train_ce)
            history["train_penalty"].append(train_penalty)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)

            should_log_epoch = (epoch + 1) % 5 == 0
            if should_log_epoch:
                logger.info(
                    "Epoch %d/%d | Train Loss %.4f | Train Acc %.4f | CE %.4f | Penalty %.4f | Val Loss %.4f | Val Acc %.4f | LRs %s",
                    epoch + 1, num_epochs, train_loss, train_acc, train_ce,
                    train_penalty, val_loss, val_acc, lr_details,
                )
            # DVCLive theo dõi đầy đủ 4 metric chính: train_loss, train_acc,
            # val_loss, val_acc — cộng thêm breakdown loss (ce/penalty) và lr.
            live.log_metric("train/loss", train_loss)
            live.log_metric("train/acc", train_acc)
            live.log_metric("train/ce", train_ce)
            live.log_metric("train/penalty", train_penalty)
            live.log_metric("val/loss", val_loss)
            live.log_metric("val/acc", val_acc)

            # live.log_metric("lr", current_lr)
            for i, group in enumerate(optimizer.param_groups):
                group_name = group.get("name", f"group_{i}")
                live.log_metric(f"lr/{group_name}", group.get("lr", 0))

             # ---- Observability: bao nhiêu luật thực sự "sống" trong epoch này ----
            if coverage_stats:
                live.log_metric("rules/coverage_ratio", coverage_stats["coverage_ratio"])
                live.log_metric("rules/mean_satisfaction", coverage_stats["mean_rule_sat"])
                if "mean_ruleset_size" in coverage_stats:
                    live.log_metric("rules/mean_ruleset_size", coverage_stats["mean_ruleset_size"])
                if should_log_epoch:
                    logger.info(
                        "  Rule coverage: %d/%d luật active (%.1f%%) | mean_rule_sat=%.4f | T=%.2f",
                        round(coverage_stats["coverage_ratio"] * coverage_stats["n_rules_total"]),
                        coverage_stats["n_rules_total"],
                        coverage_stats["coverage_ratio"] * 100,
                        coverage_stats["mean_rule_sat"],
                        penalty_module._temperature.item(),
                    )

            # Giá trị dùng để quyết định "tốt nhất" phụ thuộc monitor_metric
            # (val_acc -> mode='max', val_loss -> mode='min'), xem MONITOR_MODES.
            monitor_value = {"val_acc": val_acc, "val_loss": val_loss}[monitor_metric]

            # Một lời gọi duy nhất quyết định "đây có phải điểm tốt nhất chưa?"
            # -- vừa dùng cho quyết định lưu checkpoint, vừa cho patience counter.
            is_best = stopper(monitor_value)
            early_stop_enabled = epoch + 1 >= min_epochs_before_early_stop
            if not early_stop_enabled:
                # Vẫn theo dõi/lưu best checkpoint trong giai đoạn ủ, nhưng patience
                # chỉ bắt đầu có hiệu lực sau khi lịch nhiệt độ đã hoàn tất.
                stopper.counter = 0
                stopper.early_stop = False
            if is_best:
                best_acc = val_acc  # vẫn ghi nhận accuracy để log/báo cáo, dù đang theo dõi metric khác
                best_weights = copy.deepcopy(model.state_dict())
                save_checkpoint(
                    ckpt_name, model, optimizer, epoch + 1, best_acc,
                    class_names=getattr(train_loader.dataset, "classes", None),
                )
                live.log_metric("best_val_acc", best_acc)
                live.log_metric(f"best_{monitor_metric}", monitor_value)

            should_stop = early_stop_enabled and stopper.early_stop
            if should_stop:
                logger.info("Early stopping triggered.")
                live.log_metric("early_stop", 1)

            live.next_step()
            last_completed_epoch = epoch + 1
            if resume_enabled:
                _save_resume_checkpoint(
                    resume_path,
                    build_resume_state(
                        last_completed_epoch,
                        completed=should_stop or last_completed_epoch >= num_epochs,
                    ),
                )
            if should_stop:
                break

    elapsed = time.time() - start_time
    if penalty_module is not None:
        temp_span = final_temp - initial_temp
        progress_pct = (
            (penalty_module._temperature.item() - initial_temp) / temp_span * 100
            if abs(temp_span) > 1e-12
            else 100.0
        )
        logger.info("Nhiệt độ dừng ở %.2f (%.1f%% quãng đường ủ dự kiến)", penalty_module._temperature.item(), progress_pct)
    logger.info("Training complete in %.2f minutes | Best Val Acc: %.4f", elapsed / 60, best_acc)

    model.load_state_dict(best_weights)
    final_weight_path = os.path.join(cfg["save_dir"], "final_model_weights.pth")
    torch.save(model.state_dict(), final_weight_path)
    if resume_enabled:
        _save_resume_checkpoint(
            resume_path,
            build_resume_state(last_completed_epoch, completed=True),
        )
    return model, history
