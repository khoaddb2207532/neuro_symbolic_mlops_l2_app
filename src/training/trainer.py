"""Vòng lặp huấn luyện CNN — hỗ trợ:
  * rule-regularized loss (penalty theo luật được GFlowNet chọn) — dùng DUY
    NHẤT `VectorizedRulePenalty` (xem src/rules/penalty.py để biết lý do đã
    gộp bỏ implementation loop trùng lặp trước đây).
  * DVCLive experiment tracking
  * huấn luyện end-to-end toàn bộ model ngay từ epoch đầu tiên. Model có thể
    được khởi tạo bằng trọng số pretrained (ví dụ ImageNet), nhưng trainer
    không đóng băng layer, progressive-unfreeze hay dùng differential LR.
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
    # Nếu True: xử lý xung đột gradient (PCGrad) giữa CE loss và rule-penalty
    # loss thay vì backward trực tiếp trên tổng — xem _resolve_loss_conflict().
    # Mặc định False để KHÔNG đổi hành vi/số liệu của các pipeline hiện có.
    "resolve_loss_conflict": False,
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


def _resolve_loss_conflict(ce_loss: torch.Tensor, penalty: torch.Tensor, params: list) -> Tuple[bool, float]:
    """Xử lý xung đột gradient giữa CE loss (nhiệm vụ chính) và rule-penalty
    loss (regularizer) trên các tham số DÙNG CHUNG `params` — kỹ thuật
    PCGrad (Projecting Conflicting Gradients, Yu et al. 2020) rút gọn cho
    đúng 2 loss:

      1. Tính grad_ce, grad_penalty RIÊNG BIỆT (2 lần `torch.autograd.grad`
         trên cùng graph, không phải 2 lần forward).
      2. Nếu cos(grad_ce, grad_penalty) < 0 (2 gradient "kéo" NGƯỢC hướng
         nhau — dấu hiệu 2 loss đang xung đột trên cùng tham số) -> chiếu
         grad_penalty lên phần KHÔNG xung đột với grad_ce (loại bỏ thành
         phần đối nghịch bằng phép chiếu vector), giữ nguyên grad_ce.
      3. Nếu KHÔNG xung đột (dot >= 0) -> cộng thẳng grad_ce + grad_penalty,
         numerically giống hệt `(ce_loss + penalty).backward()` (không đổi
         hành vi cũ trong trường hợp phổ biến 2 loss "hợp tác").
      4. Gán gradient cuối cùng vào `.grad` của từng tham số (thay cho
         `loss.backward()` thông thường).

    Trả về (had_conflict, cos_similarity) để quan sát/log — không phải để
    quyết định logic bên ngoài.

    Nếu `penalty` không có graph (vd rule_set rỗng, trả về hằng số 0 không
    `requires_grad`) thì bỏ qua toàn bộ cơ chế này, chỉ backward `ce_loss`.
    """
    if not (penalty.requires_grad and penalty.grad_fn is not None):
        for p in params:
            p.grad = None
        ce_loss.backward()
        return False, 0.0

    grad_ce = torch.autograd.grad(ce_loss, params, retain_graph=True, allow_unused=True)
    grad_pen = torch.autograd.grad(penalty, params, retain_graph=False, allow_unused=True)

    flat_ce = torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grad_ce, params)
    ])
    flat_pen = torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grad_pen, params)
    ])

    dot = torch.dot(flat_ce, flat_pen)
    ce_norm_sq = flat_ce.dot(flat_ce)
    pen_norm = flat_pen.norm()
    ce_norm = ce_norm_sq.sqrt()
    cos_sim = (dot / (ce_norm * pen_norm + 1e-12)).item() if pen_norm > 0 and ce_norm > 0 else 0.0

    had_conflict = dot.item() < 0
    if had_conflict:
        # Loại bỏ thành phần của grad_penalty đối nghịch với grad_ce (chiếu
        # lên siêu phẳng trực giao với grad_ce), giữ grad_ce nguyên vẹn —
        # đúng công thức PCGrad cho cặp (task chính, regularizer).
        proj_coeff = dot / (ce_norm_sq + 1e-12)
        flat_pen = flat_pen - proj_coeff * flat_ce

    combined_flat = flat_ce + flat_pen

    idx = 0
    for p in params:
        numel = p.numel()
        p.grad = combined_flat[idx: idx + numel].view_as(p).clone()
        idx += numel

    return had_conflict, cos_sim


def train_one_epoch(
    model, loader, criterion, optimizer, device, penalty_module=None, freeze_bn=True,
    resolve_conflict: bool = False,
):
    """penalty_module: nếu không None (VectorizedRulePenalty hoặc
    BayesianRuleMarginalization — cùng chữ ký forward(features, logits)),
    được cộng vào loss chính. Không còn nhánh loop-based riêng — chỉ một
    công thức.

    resolve_conflict: nếu True và penalty_module có graph (không phải hằng
    số 0), xử lý xung đột gradient giữa CE loss và rule-penalty loss bằng
    PCGrad (xem `_resolve_loss_conflict`) thay vì backward tổng trực tiếp.
    Mặc định False để giữ nguyên hành vi cũ (loss = ce+penalty; backward
    trên tổng) cho các script chưa bật cờ này.

    Trả về (train_loss, train_ce, train_penalty, train_acc, coverage_stats).
    train_acc được tính trên chính batch train (không phải eval mode) để
    theo dõi cùng lúc với val_acc trên DVCLive, giúp phát hiện overfit/
    underfit qua khoảng cách train_acc vs val_acc.

    coverage_stats (dict rỗng nếu không có penalty_module hoặc module không
    hỗ trợ `last_coverage_stats()`): thống kê xem luật có thực sự "sống"
    (đóng góp gradient) hay không, tổng hợp theo batch-size-weighted average
    qua các batch trong epoch — xem src/rules/penalty.py::last_coverage_stats().
    Khi resolve_conflict=True, coverage_stats còn thêm "conflict_ratio":
    tỉ lệ batch mà CE và rule-penalty xung đột (cos<0) trong epoch này.
    """
    model.train()
    if freeze_bn and hasattr(model, "freeze_bn"):
        model.freeze_bn()

    total_loss = total_ce = total_penalty = 0.0
    total_correct = 0
    total_samples = 0
    n_conflict_batches = 0
    n_batches = 0

    # Tích luỹ coverage stats qua các batch (weighted theo batch size, giống
    # cách total_loss/total_ce/... được tích luỹ ở trên).
    sum_coverage_ratio = 0.0
    sum_mean_sat = 0.0
    n_rules_total = 0
    has_coverage = penalty_module is not None and hasattr(penalty_module, "last_coverage_stats")

    trainable_params = [p for p in model.parameters() if p.requires_grad] if resolve_conflict else None

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        outputs = model(images)
        logits, features = outputs if isinstance(outputs, (tuple, list)) and len(outputs) == 2 else (outputs, None)

        ce_loss = criterion(logits, labels)
        if penalty_module is None:
            penalty = torch.tensor(0.0, device=device)
        elif getattr(penalty_module, "requires_labels", False):
            penalty = penalty_module(features, logits, labels)
        else:
            penalty = penalty_module(features, logits)

        if resolve_conflict and penalty_module is not None:
            had_conflict, _cos_sim = _resolve_loss_conflict(ce_loss, penalty, trainable_params)
            n_conflict_batches += int(had_conflict)
            n_batches += 1
        else:
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
        if resolve_conflict:
            coverage_stats["conflict_ratio"] = n_conflict_batches / n_batches if n_batches > 0 else 0.0

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

    # Normal end-to-end training: mọi tham số, kể cả backbone pretrained, đều
    # được cập nhật ngay từ epoch đầu tiên.
    if hasattr(model, "set_freeze_stage"):
        model.set_freeze_stage("full")
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True

    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    is_rule_regularized_run = penalty_weight > 0 or penalty_module is not None
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
        ).to(device)
    elif penalty_module is not None:
        penalty_module = penalty_module.to(device)
        logger.info("Dùng penalty_module được truyền sẵn: %s", type(penalty_module).__name__)

    resolve_conflict = cfg.get("resolve_loss_conflict", False) and penalty_module is not None
    if resolve_conflict:
        logger.info("Bật xử lý xung đột gradient (PCGrad) giữa CE loss và rule-penalty loss.")

    penalty_warmup_start = int(cfg.get("penalty_warmup_start_epoch", 0))
    penalty_warmup_end = int(cfg.get("penalty_warmup_end_epoch", penalty_warmup_start))
    temperature_end_epoch = int(cfg.get("temperature_end_epoch", num_epochs - 1))
    intermediate_temp = cfg.get("intermediate_temp")
    full_stage_epoch = 0
    if penalty_warmup_end < penalty_warmup_start:
        raise ValueError("penalty_warmup_end_epoch phải >= penalty_warmup_start_epoch")

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
                "num_epochs": num_epochs, "lr": cfg["lr"], "patience": patience,
                "penalty_weight": penalty_weight, "smoothing": smoothing, "num_classes": num_classes,
                "training_mode": "end_to_end",
                "batch_size": train_loader.batch_size,
                "total_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "initial_temp": initial_temp, 
                "final_temp": final_temp,
                "intermediate_temp": intermediate_temp,
                "penalty_warmup_start_epoch": penalty_warmup_start,
                "penalty_warmup_end_epoch": penalty_warmup_end,
                "temperature_end_epoch": temperature_end_epoch,
                "resolve_loss_conflict": resolve_conflict,
            }
        )

        for epoch in range(num_epochs):
            # ---- Warm-up penalty: ưu tiên CE khi representation còn chưa ổn
            # định, rồi tăng lambda tuyến tính tới penalty_weight cấu hình. ----
            if penalty_module is not None:
                if epoch < penalty_warmup_start:
                    current_penalty_weight = 0.0
                elif penalty_warmup_end == penalty_warmup_start or epoch >= penalty_warmup_end:
                    current_penalty_weight = penalty_weight
                else:
                    warmup_progress = (
                        (epoch - penalty_warmup_start)
                        / (penalty_warmup_end - penalty_warmup_start)
                    )
                    current_penalty_weight = penalty_weight * warmup_progress
                penalty_module.penalty_weight = current_penalty_weight

                # Hai pha ủ T nếu có intermediate_temp: initial -> intermediate
                # trước stage full, rồi intermediate -> final tới mốc cấu hình.
                if intermediate_temp is not None and epoch <= full_stage_epoch:
                    progress = epoch / max(full_stage_epoch, 1)
                    current_temp = initial_temp + (float(intermediate_temp) - initial_temp) * progress
                elif intermediate_temp is not None:
                    progress = min(
                        (epoch - full_stage_epoch) / max(temperature_end_epoch - full_stage_epoch, 1),
                        1.0,
                    )
                    current_temp = float(intermediate_temp) + (final_temp - float(intermediate_temp)) * progress
                else:
                    progress = min(epoch / max(temperature_end_epoch, 1), 1.0)
                    current_temp = initial_temp + (final_temp - initial_temp) * progress
                penalty_module._temperature.fill_(current_temp)

                live.log_metric("rules/penalty_weight", current_penalty_weight)
                live.log_metric("rules/temperature", penalty_module._temperature.item())

            train_loss, train_ce, train_penalty, train_acc, coverage_stats = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                penalty_module, freeze_bn=True,
                resolve_conflict=resolve_conflict,
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

            if scheduler is not None:
                scheduler.step(val_loss)

            logger.info(
                "Epoch %d/%d | Train Loss %.4f | Train Acc %.4f | CE %.4f | Penalty %.4f | Val Loss %.4f | Val Acc %.4f | LRs %s ",
                epoch + 1, num_epochs, train_loss, train_acc, train_ce, train_penalty, val_loss, val_acc, lr_details,
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
                if "conflict_ratio" in coverage_stats:
                    live.log_metric("rules/grad_conflict_ratio", coverage_stats["conflict_ratio"])
                logger.info(
                    "  Rule coverage: %d/%d luật active (%.1f%%) | mean_rule_sat=%.4f | lambda=%.4f | T=%.2f%s",
                    round(coverage_stats["coverage_ratio"] * coverage_stats["n_rules_total"]),
                    coverage_stats["n_rules_total"],
                    coverage_stats["coverage_ratio"] * 100,
                    coverage_stats["mean_rule_sat"],
                    penalty_module.penalty_weight,
                    penalty_module._temperature.item(),
                    f" | grad_conflict_ratio={coverage_stats['conflict_ratio']:.2f}" if "conflict_ratio" in coverage_stats else "",
                )    

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
    if penalty_module is not None:
        progress_pct = (penalty_module._temperature.item() - initial_temp) / (final_temp - initial_temp) * 100
        logger.info("Nhiệt độ dừng ở %.2f (%.1f%% quãng đường ủ dự kiến)", penalty_module._temperature.item(), progress_pct)
    logger.info("Training complete in %.2f minutes | Best Val Acc: %.4f", elapsed / 60, best_acc)

    model.load_state_dict(best_weights)
    final_weight_path = os.path.join(cfg["save_dir"], "final_model_weights.pth")
    torch.save(model.state_dict(), final_weight_path)
    return model, history
