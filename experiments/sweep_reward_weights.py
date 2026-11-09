"""
Sweep hyperparameter 3 giai đoạn cho GFlowNet rule selection, dùng Optuna.

KHÔNG phải DVC/pipeline stage chính thức — đây là script thử nghiệm để TÌM
bộ tham số tốt, sau đó ghi ngược vào params.yaml["gflownet"] để stage4 đọc.

Chạy:
    python -m experiments.sweep_reward_weights --config /kaggle/working/params.yaml
Sau đó:
    python -m pipelines.stage4_select_rules_gflownet --config /kaggle/working/params.yaml
"""
import argparse
import csv
import json
import os
import pickle
import statistics
from typing import Dict, List, Tuple
import shutil
import optuna
import torch
import yaml

from src.gflownet.pipeline import RuleExtractionPipeline
from src.gflownet.evaluation import evaluate_run
from src.rules.rule_types import Rule
from src.rules.validator import RuleValidator
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)  # tránh log Optuna đè log của mình


# --------------------------------------------------------------------------
# 0. Load dữ liệu MỘT LẦN — dùng chung cho toàn bộ 3 giai đoạn, mọi trial
# --------------------------------------------------------------------------
def load_common_data(params: dict, device: str) -> Tuple[
    List[Rule], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    features_dir = os.path.join(params["output_dir"], "02_features")
    rules_dir = os.path.join(params["output_dir"], "03_rules")

    with open(os.path.join(rules_dir, "raw_rules.pkl"), "rb") as f:
        raw_rules = pickle.load(f)

    val_features = torch.load(f"{features_dir}/val_features.pt").to(device)
    val_labels = torch.load(f"{features_dir}/val_labels.pt").to(device)

    validator = RuleValidator(
        min_supp=params["rules"]["min_support"],
        min_conf=params["rules"]["min_confidence"],
    )
    valid_rule_set, cover, correct, rule_len = validator.validate_and_build_tensors(
        raw_rules, val_features, val_labels, store_device=device
    )
    valid_rules = list(valid_rule_set.rules)
    logger.info("Đã load %d luật hợp lệ (dùng chung cho toàn bộ sweep).", len(valid_rules))
    return valid_rules, cover, correct, rule_len, val_labels


# --------------------------------------------------------------------------
# Hàm chạy 1 lần train+eval, dùng chung cho cả 3 giai đoạn
# --------------------------------------------------------------------------
def run_one_trial(
    cfg: Dict,
    valid_rules: List[Rule],
    cover: torch.Tensor,
    correct: torch.Tensor,
    rule_len: torch.Tensor,
    labels: torch.Tensor,
    max_rules: int,
    num_iterations: int,
    device: str,
    output_dir: str,
    fixed_train_kwargs: Dict,
) -> Dict:
    pipeline = RuleExtractionPipeline(
        device=device,
        w_acc=cfg["w_acc"], w_cov=cfg["w_cov"],
        w_wrong=cfg.get("w_wrong", 0.75),
        w_conflict=cfg["w_conflict"],
        beta=cfg["beta"],
    )
    selected = pipeline.run(
        valid_rules=valid_rules, cover=cover, correct=correct, rule_len=rule_len,
        labels=labels,
        max_rules=max_rules, output_dir=output_dir,
        num_iterations=num_iterations,
        device=device,
        **fixed_train_kwargs,
    )
    reward_module = getattr(pipeline, "_last_reward_module", None)
    metric = evaluate_run(selected, valid_rules, reward_module)
    metric.update(cfg)
    metric["max_rules"] = max_rules
    shutil.rmtree(output_dir, ignore_errors=True)
    return metric


# --------------------------------------------------------------------------
# Chọn 1 điểm duy nhất trên Pareto front — CHỈ dùng SAU KHI search đa mục tiêu
# đã xong, KHÔNG dùng để lái quá trình search (đó là việc của NSGA-II).
#
# Vì sao tách riêng "search" và "chọn điểm" thành 2 bước:
#   - Nếu dùng 1 công thức cố định (weighted sum) để LÀM OBJECTIVE cho Optuna
#     ngay từ đầu, ta quay lại đúng vấn đề của f1_like: một số chiều mục tiêu
#     bị bỏ qua hoặc bị thiên lệch bởi chính tham số đang sweep.
#   - Multi-objective (NSGA-II) khám phá ĐỀU trên toàn bộ không gian đánh đổi
#     3 chiều (accuracy, coverage, redundancy_conflict), không thiên vị.
#   - Sau khi có Pareto front (nhiều lựa chọn không ai thống trị ai), NGƯỜI
#     NGHIÊN CỨU mới áp "khẩu vị" của mình (lambda_conflict) để chọn RA 1
#     điểm — vì pipeline sau (stage2, stage3, ghi params.yaml) cần 1 bộ
#     trọng số duy nhất, không thể mang cả 1 tập nghiệm đi tiếp.
#   - lambda_conflict ở đây là hằng số CỐ ĐỊNH do người dùng chọn, hoàn toàn
#     tách biệt khỏi w_conflict (là biến được GFlowNet sweep) — không có
#     chuyện tự thưởng cho chính tham số đang được tối ưu.
# --------------------------------------------------------------------------
def select_from_pareto_front(
    study: "optuna.Study",
    lambda_conflict: float = 0.3,
) -> "optuna.trial.FrozenTrial":
    pareto_trials = study.best_trials
    if not pareto_trials:
        raise RuntimeError(
            f"Study '{study.study_name}' không có trial nào trên Pareto front — "
            "kiểm tra lại đã optimize() ít nhất 1 trial thành công chưa."
        )
    if len(pareto_trials) < 3:
        logger.warning(
            "Pareto front của '%s' chỉ có %d điểm — có thể do n_trials quá ít "
            "cho NSGA-II (khuyến nghị >=30 trial). Cân nhắc tăng n_trials.",
            study.study_name, len(pareto_trials),
        )

    def scalarize(t: "optuna.trial.FrozenTrial") -> float:
        accuracy, coverage, redundancy_conflict = t.values
        return accuracy + coverage - lambda_conflict * redundancy_conflict

    chosen = max(pareto_trials, key=scalarize)
    logger.info(
        "[%s] Pareto front: %d điểm. Chọn trial=%d (lambda_conflict=%.2f) "
        "-> (accuracy,coverage,redundancy_conflict)=%s, params=%s",
        study.study_name, len(pareto_trials), chosen.number,
        lambda_conflict, chosen.values, chosen.params,
    )
    return chosen


# --------------------------------------------------------------------------
# GIAI ĐOẠN 1 — sweep w_acc / w_cov / w_conflict (reward shape)
# beta và max_rules giữ cố định ở giá trị mặc định từ params.yaml
#
# ĐÃ BỎ w_red (trùng lặp cùng target) và w_comp (số lượng luật) khỏi công
# thức reward — reward giờ CHỈ phục vụ mục đích chọn luật tốt cho
# regularization CNN, không phải để tối ưu tính gọn/dễ đọc của tập luật (đó
# là thống kê mô tả riêng, không phải mục tiêu GFlowNet tối ưu). Vì vậy
# objective đa mục tiêu giờ CHỈ còn 3 chiều: (accuracy, coverage,
# redundancy_conflict) — bỏ complexity hoàn toàn khỏi cả search lẫn lựa chọn
# điểm trên Pareto front. Xem thảo luận & reward.py để biết lý do đầy đủ.
# --------------------------------------------------------------------------
def stage1_reward_weights(
    params: dict, valid_rules, cover, correct, rule_len, labels, device: str,
    n_trials: int, storage_path: str, sweep_root: str,
    pareto_lambda_conflict: float = 0.3,
) -> Dict:
    gfn_cfg = params["gflownet"]
    max_rules = gfn_cfg["max_rules"]
    fixed_kwargs = dict(
        gfnet_hidden_dim=gfn_cfg["hidden_dim"],
        batch_size=gfn_cfg["batch_size"],
        lr=gfn_cfg["lr"], logZ_lr=gfn_cfg["logZ_lr"],
        validation_interval=100, loss_type=gfn_cfg["loss_type"],
        logZ_warmup_steps=50, val_samples=10,
    )
    num_iterations = 800  # rút gọn cho sweep, không cần full iteration ở giai đoạn dò

    def objective(trial: optuna.Trial):
        cfg = {
            "w_acc": trial.suggest_float("w_acc", 0.7, 1.3),
            "w_cov": trial.suggest_float("w_cov", 0.2, 0.8),
            "w_wrong": 0.75,
            "w_conflict": trial.suggest_float("w_conflict", 0.0, 0.3),
            "beta": 3.0,  # cố định ở giai đoạn 1
        }
        set_seed(params["seed"] + trial.number)
        output_dir = os.path.join(sweep_root, "stage1", f"trial_{trial.number}")
        os.makedirs(output_dir, exist_ok=True)

        metric = run_one_trial(cfg, valid_rules, cover, correct, rule_len, labels,
                                max_rules, num_iterations, device, output_dir, fixed_kwargs)
        for k, v in metric.items():
            trial.set_user_attr(k, v)
        logger.info("[Stage1] Trial %d: %s", trial.number, metric)
        # 3 mục tiêu riêng biệt (đã bỏ complexity) — KHÔNG gộp thành 1 scalar
        # ở đây (xem docstring select_from_pareto_front phía trên vì sao).
        return metric["accuracy"], metric["coverage"], metric["redundancy_conflict"]

    study = optuna.create_study(
        study_name="stage1_reward_weights",
        directions=["maximize", "maximize", "minimize"],
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(seed=params["seed"]),
    )
    remaining = n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    else:
        logger.info("[Stage1] Đã đủ %d trials từ lần chạy trước, bỏ qua.", n_trials)

    chosen = select_from_pareto_front(study, pareto_lambda_conflict)
    return {**chosen.params, "beta": 3.0}  # trả về w_acc/w_cov/w_conflict đã chọn từ Pareto front


# --------------------------------------------------------------------------
# GIAI ĐOẠN 2 — sweep beta + max_rules, dùng w_* tốt nhất từ giai đoạn 1
# --------------------------------------------------------------------------
def stage2_beta_maxrules(
    params: dict, best_weights: Dict, valid_rules, cover, correct, rule_len,
    labels, device: str,
    n_trials: int, storage_path: str, sweep_root: str,
    pareto_lambda_conflict: float = 0.3,
) -> Dict:
    gfn_cfg = params["gflownet"]
    fixed_kwargs = dict(
        gfnet_hidden_dim=gfn_cfg["hidden_dim"],
        batch_size=gfn_cfg["batch_size"],
        lr=gfn_cfg["lr"], logZ_lr=gfn_cfg["logZ_lr"],
        validation_interval=100, loss_type=gfn_cfg["loss_type"],
        logZ_warmup_steps=50, val_samples=10,
    )
    num_iterations = 800

    def objective(trial: optuna.Trial):
        cfg = {
            "w_acc": best_weights["w_acc"], "w_cov": best_weights["w_cov"],
            "w_wrong": best_weights.get("w_wrong", 0.75),
            "w_conflict": best_weights["w_conflict"],
            "beta": trial.suggest_float("beta", 1.0, 8.0),
        }
        max_rules = trial.suggest_categorical("max_rules", [16, 24, 32, 40, 48])
        set_seed(params["seed"] + trial.number)
        output_dir = os.path.join(sweep_root, "stage2", f"trial_{trial.number}")
        os.makedirs(output_dir, exist_ok=True)

        metric = run_one_trial(cfg, valid_rules, cover, correct, rule_len, labels,
                                max_rules, num_iterations, device, output_dir, fixed_kwargs)
        for k, v in metric.items():
            trial.set_user_attr(k, v)
        logger.info("[Stage2] Trial %d: %s", trial.number, metric)
        # max_rules giờ CHỈ là trần ngân sách, không còn bị phạt complexity —
        # ảnh hưởng của nó lên accuracy/coverage/redundancy_conflict là tín
        # hiệu THẬT (ví dụ max_rules lớn có thể làm redundancy_conflict tăng
        # tự nhiên do dễ lẫn luật xung đột hơn), không còn bị thiên lệch nhân
        # tạo như hồi dùng f1_like.
        return metric["accuracy"], metric["coverage"], metric["redundancy_conflict"]

    study = optuna.create_study(
        study_name="stage2_beta_maxrules",
        directions=["maximize", "maximize", "minimize"],
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(seed=params["seed"]),
    )
    remaining = n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    else:
        logger.info("[Stage2] Đã đủ %d trials từ lần chạy trước, bỏ qua.", n_trials)

    chosen = select_from_pareto_front(study, pareto_lambda_conflict)
    return {"beta": chosen.params["beta"], "max_rules": chosen.params["max_rules"]}


# --------------------------------------------------------------------------
# GIAI ĐOẠN 3 — sweep lr/batch_size, dùng reward + beta + max_rules tốt nhất
# từ giai đoạn 1 và 2. Chạy full num_iterations vì mục đích là xác nhận
# hội tụ, không phải dò nhanh như 2 giai đoạn trước.
#
# NHẤT QUÁN với stage1/2: objective vẫn đa mục tiêu, giờ CHỈ còn 3 chiều
# (accuracy, coverage, redundancy_conflict) — đã bỏ complexity khỏi toàn bộ
# 3 giai đoạn, không riêng gì stage1/2.
# --------------------------------------------------------------------------
def stage3_training_dynamics(
    params: dict, best_weights: Dict, best_beta_maxrules: Dict,
    valid_rules, cover, correct, rule_len, labels, device: str,
    n_trials: int, storage_path: str, sweep_root: str,
    pareto_lambda_conflict: float = 0.3,
) -> Dict:
    gfn_cfg = params["gflownet"]
    max_rules = best_beta_maxrules["max_rules"]
    num_iterations = gfn_cfg["num_iterations"]  # full, không rút gọn ở giai đoạn cuối

    def objective(trial: optuna.Trial):
        cfg = {
            "w_acc": best_weights["w_acc"], "w_cov": best_weights["w_cov"],
            "w_wrong": best_weights.get("w_wrong", 0.75),
            "w_conflict": best_weights["w_conflict"],
            "beta": best_beta_maxrules["beta"],
        }
        lr = trial.suggest_categorical("lr", [5e-4, 1e-3, 2e-3])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        fixed_kwargs = dict(
            gfnet_hidden_dim=gfn_cfg["hidden_dim"],
            batch_size=batch_size,
            lr=lr, logZ_lr=gfn_cfg["logZ_lr"],
            validation_interval=100, loss_type=gfn_cfg["loss_type"],
            logZ_warmup_steps=50, val_samples=10,
        )

        set_seed(params["seed"] + trial.number)
        output_dir = os.path.join(sweep_root, "stage3", f"trial_{trial.number}")
        os.makedirs(output_dir, exist_ok=True)

        metric = run_one_trial(cfg, valid_rules, cover, correct, rule_len, labels,
                                max_rules, num_iterations, device, output_dir, fixed_kwargs)
        metric["lr"] = lr
        metric["batch_size"] = batch_size
        for k, v in metric.items():
            trial.set_user_attr(k, v)
        logger.info("[Stage3] Trial %d: %s", trial.number, metric)
        return metric["accuracy"], metric["coverage"], metric["redundancy_conflict"]

    study = optuna.create_study(
        study_name="stage3_training_dynamics",
        directions=["maximize", "maximize", "minimize"],
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(seed=params["seed"]),
    )
    remaining = n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    else:
        logger.info("[Stage3] Đã đủ %d trials từ lần chạy trước, bỏ qua.", n_trials)

    chosen = select_from_pareto_front(study, pareto_lambda_conflict)
    return {"lr": chosen.params["lr"], "batch_size": chosen.params["batch_size"]}


# --------------------------------------------------------------------------
# XÁC NHẬN ỔN ĐỊNH — sau khi đã CHỌN xong 1 bộ tham số cuối cùng (best_weights +
# best_beta_maxrules + best_training), chạy lại TOÀN BỘ pipeline GFlowNet với
# CÙNG 1 cấu hình nhưng NHIỀU SEED khác nhau, lấy mean±std cho tất cả metric.
#
# Vì sao cần bước này (đã thảo luận trước): huấn luyện GFlowNet không tất định
# (khởi tạo mạng ngẫu nhiên + sampling trajectory ngẫu nhiên trong trajectory
# balance) — 1 lần chạy không đủ căn cứ để báo cáo trong luận văn. Đây KHÔNG
# phải sweep tìm tham số (không có trial.suggest_* nào), chỉ là lặp lại đúng
# 1 cấu hình đã chốt để đo phương sai do ngẫu nhiên huấn luyện gây ra.
# --------------------------------------------------------------------------
def multi_seed_validation(
    params: dict,
    best_weights: Dict,
    best_beta_maxrules: Dict,
    best_training: Dict,
    valid_rules, cover, correct, rule_len, labels,
    device: str,
    n_seeds: int,
    sweep_root: str,
) -> Dict:
    if n_seeds < 5:
        logger.warning(
            "n_seeds=%d < 5 — số seed thấp có thể cho mean/std không đáng tin. "
            "Khuyến nghị >=5 (lý tưởng 8-10 nếu đủ thời gian).", n_seeds,
        )

    gfn_cfg = params["gflownet"]
    max_rules = best_beta_maxrules["max_rules"]
    num_iterations = gfn_cfg["num_iterations"]  # full, giống lúc train thật ở stage4 chính thức

    cfg = {
        "w_acc": best_weights["w_acc"], "w_cov": best_weights["w_cov"],
        "w_wrong": best_weights.get("w_wrong", 0.75),
        "w_conflict": best_weights["w_conflict"],
        "beta": best_beta_maxrules["beta"],
    }
    fixed_kwargs = dict(
        gfnet_hidden_dim=gfn_cfg["hidden_dim"],
        batch_size=best_training["batch_size"],
        lr=best_training["lr"], logZ_lr=gfn_cfg["logZ_lr"],
        validation_interval=100, loss_type=gfn_cfg["loss_type"],
        logZ_warmup_steps=50, val_samples=10,
    )

    # Offset seed hẳn ra khỏi vùng seed đã dùng trong 3 giai đoạn sweep phía
    # trên (params["seed"] + trial.number, trial.number thường < n_trials sweep)
    # để không vô tình lặp lại đúng 1 seed đã thấy trong lúc tìm tham số.
    base_seed = params["seed"] + 10_000
    seeds = [base_seed + i for i in range(n_seeds)]

    # "redundancy"/"complexity" vẫn được BÁO CÁO (thống kê mô tả, dùng cho
    # phần diễn giải tập luật), dù không còn là mục tiêu GFlowNet tối ưu.
    # "redundancy_conflict" mới là thành phần thực sự nằm trong reward.
    metric_keys = ["n_rules", "accuracy", "coverage", "redundancy", "redundancy_conflict", "complexity", "f1_like"]
    records = []
    for i, seed in enumerate(seeds):
        set_seed(seed)
        output_dir = os.path.join(sweep_root, "multi_seed_validation", f"seed_{seed}")
        os.makedirs(output_dir, exist_ok=True)
        metric = run_one_trial(cfg, valid_rules, cover, correct, rule_len, labels,
                                max_rules, num_iterations, device, output_dir, fixed_kwargs)
        metric["seed"] = seed
        records.append(metric)
        logger.info("[MultiSeedValidation] (%d/%d) seed=%d: %s", i + 1, n_seeds, seed, metric)

    summary_stats = {}
    for k in metric_keys:
        vals = [r[k] for r in records]
        summary_stats[f"{k}_mean"] = statistics.mean(vals)
        summary_stats[f"{k}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    logger.info("[MultiSeedValidation] Mean/Std trên %d seed: %s", n_seeds, summary_stats)

    # ---- Lưu chi tiết từng seed (để tự kiểm tra outlier nếu cần) ----
    detail_csv_path = os.path.join(sweep_root, "gflownet_multiseed_detail.csv")
    with open(detail_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed"] + metric_keys)
        writer.writeheader()
        for r in records:
            writer.writerow({"seed": r["seed"], **{k: r[k] for k in metric_keys}})

    # ---- Lưu 1 dòng tổng hợp mean±std, ĐẶT TÊN CỘT KHỚP với CSV so sánh
    # heuristic trước đó (method, so_luat, so_luat_std, accuracy, accuracy_std,
    # coverage, coverage_std, redundancy, redundancy_std, f1_like, f1_like_std,
    # n_runs) để bạn có thể NỐI TRỰC TIẾP dòng này vào file so sánh cũ.
    # Thêm redundancy_conflict/redundancy_conflict_std (thành phần thực sự
    # nằm trong reward giờ đây) — CSV so sánh heuristic cũ (random/topk/greedy)
    # sẽ KHÔNG có 2 cột này (vì heuristic không dùng reward này), để trống
    # hoặc bạn tự bổ sung nếu muốn so sánh chéo.
    summary_csv_path = os.path.join(sweep_root, "gflownet_multiseed_summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method", "budget", "budget_value",
            "so_luat", "so_luat_std",
            "accuracy", "accuracy_std",
            "coverage", "coverage_std",
            "redundancy", "redundancy_std",
            "redundancy_conflict", "redundancy_conflict_std",
            "complexity", "complexity_std",
            "f1_like", "f1_like_std",
            "n_runs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "method": "gflownet_optimized",
            "budget": "self",
            "budget_value": max_rules,
            "so_luat": summary_stats["n_rules_mean"],
            "so_luat_std": summary_stats["n_rules_std"],
            "accuracy": summary_stats["accuracy_mean"],
            "accuracy_std": summary_stats["accuracy_std"],
            "coverage": summary_stats["coverage_mean"],
            "coverage_std": summary_stats["coverage_std"],
            "redundancy": summary_stats["redundancy_mean"],
            "redundancy_std": summary_stats["redundancy_std"],
            "redundancy_conflict": summary_stats["redundancy_conflict_mean"],
            "redundancy_conflict_std": summary_stats["redundancy_conflict_std"],
            "complexity": summary_stats["complexity_mean"],
            "complexity_std": summary_stats["complexity_std"],
            "f1_like": summary_stats["f1_like_mean"],
            "f1_like_std": summary_stats["f1_like_std"],
            "n_runs": n_seeds,
        })

    logger.info(
        "[MultiSeedValidation] Đã lưu: chi tiết -> %s | tổng hợp (1 dòng, sẵn để "
        "nối vào bảng so sánh heuristic) -> %s", detail_csv_path, summary_csv_path,
    )

    return {
        "seeds": seeds,
        "per_seed_records": records,
        "summary_stats": summary_stats,
        "detail_csv_path": detail_csv_path,
        "summary_csv_path": summary_csv_path,
    }


# --------------------------------------------------------------------------
# Ghi bộ tham số tốt nhất ngược lại vào params.yaml, để stage4 đọc trực tiếp
# --------------------------------------------------------------------------
def update_params_yaml(
    params_path: str,
    best_weights: Dict,
    best_beta_maxrules: Dict,
    best_training: Dict,
) -> None:
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    gfn_cfg = params.setdefault("gflownet", {})
    gfn_cfg["w_acc"] = best_weights["w_acc"]
    gfn_cfg["w_cov"] = best_weights["w_cov"]
    gfn_cfg["w_wrong"] = best_weights.get("w_wrong", 0.75)
    gfn_cfg["w_conflict"] = best_weights["w_conflict"]
    # Dọn key cũ (w_red/w_comp) nếu params.yaml từ lần chạy trước còn sót lại,
    # tránh nhầm lẫn khi đọc file thấy cả key cũ lẫn mới.
    gfn_cfg.pop("w_red", None)
    gfn_cfg.pop("w_comp", None)
    gfn_cfg["beta"] = best_beta_maxrules["beta"]
    gfn_cfg["max_rules"] = best_beta_maxrules["max_rules"]
    gfn_cfg["lr"] = best_training["lr"]
    gfn_cfg["batch_size"] = best_training["batch_size"]

    # Backup file gốc trước khi ghi đè, phòng khi cần đối chiếu lại
    backup_path = params_path + ".bak"
    if not os.path.exists(backup_path):
        with open(backup_path, "w") as f:
            yaml.safe_dump(params, f, allow_unicode=True, sort_keys=False)

    with open(params_path, "w") as f:
        yaml.safe_dump(params, f, allow_unicode=True, sort_keys=False)

    logger.info("Đã cập nhật %s với bộ tham số tối ưu:", params_path)
    logger.info(json.dumps(gfn_cfg, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------
# Chạy RIÊNG bước xác nhận đa-seed, dùng bộ tham số đã chốt sẵn trong
# sweep_summary.json — KHÔNG đụng tới stage1/2/3, không cần Optuna storage.
# Dùng khi: đã sweep xong từ trước, giờ chỉ muốn chạy lại validation (ví dụ
# tăng n_seeds, hoặc lần trước lỗi giữa chừng).
# --------------------------------------------------------------------------
def _run_validation_only(params_path: str, n_seeds_validation: int) -> Dict:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sweep_root = os.path.join(params["output_dir"], "experiments")
    summary_path = os.path.join(sweep_root, "sweep_summary.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Không tìm thấy {summary_path}. --only_multiseed_validation cần đã chạy "
            "sweep đầy đủ ít nhất 1 lần trước đó (để có best_weights/best_beta_maxrules/"
            "best_training). Chạy `python -m experiments.sweep_reward_weights --config "
            f"{params_path}` (không có --only_multiseed_validation) trước."
        )

    with open(summary_path, "r") as f:
        summary = json.load(f)
    best_weights = summary["best_weights"]
    best_beta_maxrules = summary["best_beta_maxrules"]
    best_training = summary["best_training"]
    logger.info(
        "[ValidationOnly] Dùng bộ tham số có sẵn từ %s: weights=%s, beta_maxrules=%s, training=%s",
        summary_path, best_weights, best_beta_maxrules, best_training,
    )

    valid_rules, cover, correct, rule_len, labels = load_common_data(params, device)

    validation_result = multi_seed_validation(
        params, best_weights, best_beta_maxrules, best_training,
        valid_rules, cover, correct, rule_len, labels, device,
        n_seeds=n_seeds_validation, sweep_root=sweep_root,
    )

    summary["multi_seed_validation"] = {
        "summary_stats": validation_result["summary_stats"],
        "seeds": validation_result["seeds"],
        "detail_csv_path": validation_result["detail_csv_path"],
        "summary_csv_path": validation_result["summary_csv_path"],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("[ValidationOnly] Hoàn tất, đã cập nhật %s", summary_path)
    return summary


# --------------------------------------------------------------------------
# Main — chạy tuần tự 3 giai đoạn
# --------------------------------------------------------------------------
def main(params_path: str, n_trials_stage1: int, n_trials_stage2: int, n_trials_stage3: int,
          force_resweep: bool = False,
          pareto_lambda_conflict: float = 0.3,
          n_seeds_validation: int = 5, skip_multiseed_validation: bool = False):
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sweep_root = os.path.join(params["output_dir"],"experiments")
    os.makedirs(sweep_root, exist_ok=True)
    storage_path = os.path.join(sweep_root, "optuna_study.db")
    summary_path = os.path.join(sweep_root, "sweep_summary.json")

    # Load 1 lần, dùng chung cho cả sweep VÀ multi-seed validation phía dưới
    # (kể cả khi tái sử dụng summary cũ, vẫn cần valid_rules/cover/correct/rule_len
    # để chạy lại multi-seed validation).
    valid_rules, cover, correct, rule_len, labels = load_common_data(params, device)

    # --- Nếu đã có kết quả sweep trước đó và không ép chạy lại -> tái sử dụng ---
    if os.path.exists(summary_path) and not force_resweep:
        logger.info("Đã tìm thấy kết quả sweep trước đó tại %s — TÁI SỬ DỤNG, không chạy lại Optuna.", summary_path)
        with open(summary_path, "r") as f:
            summary = json.load(f)
        best_weights = summary["best_weights"]
        best_beta_maxrules = summary["best_beta_maxrules"]
        best_training = summary["best_training"]
        update_params_yaml(params_path, best_weights, best_beta_maxrules, best_training)
        logger.info("Đã nạp lại bộ tham số cũ vào %s. Dùng --force_resweep nếu muốn sweep lại từ đầu.", params_path)
    else:
        # --- Ngược lại: chạy sweep như bình thường ---
        if n_trials_stage1 < 30 or n_trials_stage2 < 30 or n_trials_stage3 < 20:
            logger.warning(
                "n_trials_stage1=%d, n_trials_stage2=%d, n_trials_stage3=%d khá thấp cho "
                "multi-objective (NSGA-II) — front có thể chưa hội tụ tốt. Khuyến nghị "
                ">=30 cho stage1/2 và >=20 cho stage3 (stage3 chạy full num_iterations "
                "nên tốn thời gian hơn nhiều mỗi trial — cân nhắc trade-off).",
                n_trials_stage1, n_trials_stage2, n_trials_stage3,
            )

        logger.info("GIAI ĐOẠN 1: Sweep reward weights (multi-objective)")
        best_weights = stage1_reward_weights(
            params, valid_rules, cover, correct, rule_len, labels, device,
            n_trials=n_trials_stage1, storage_path=storage_path, sweep_root=sweep_root,
            pareto_lambda_conflict=pareto_lambda_conflict,
        )

        logger.info("GIAI ĐOẠN 2: Sweep beta + max_rules (multi-objective)")
        best_beta_maxrules = stage2_beta_maxrules(
            params, best_weights, valid_rules, cover, correct, rule_len, labels, device,
            n_trials=n_trials_stage2, storage_path=storage_path, sweep_root=sweep_root,
            pareto_lambda_conflict=pareto_lambda_conflict,
        )

        logger.info("GIAI ĐOẠN 3: Sweep lr/batch_size (multi-objective)")
        best_training = stage3_training_dynamics(
            params, best_weights, best_beta_maxrules,
            valid_rules, cover, correct, rule_len, labels, device,
            n_trials=n_trials_stage3, storage_path=storage_path, sweep_root=sweep_root,
            pareto_lambda_conflict=pareto_lambda_conflict,
        )

        update_params_yaml(params_path, best_weights, best_beta_maxrules, best_training)

        summary = {
            "best_weights": best_weights,
            "best_beta_maxrules": best_beta_maxrules,
            "best_training": best_training,
            # Ghi lại để tái lập đúng lựa chọn (đây là "khẩu vị" cố định dùng để
            # chọn 1 điểm trên Pareto front, KHÔNG phải tham số được sweep).
            "pareto_selection": {
                "lambda_conflict": pareto_lambda_conflict,
            },
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Sweep hoàn tất, kết quả lưu tại %s để tái sử dụng về sau.", summary_path)

    # --- XÁC NHẬN ỔN ĐỊNH: chạy lại đúng bộ tham số đã chọn với nhiều seed ---
    if skip_multiseed_validation:
        logger.info("Bỏ qua multi-seed validation theo yêu cầu (--skip_multiseed_validation).")
        return summary

    logger.info(
        "XÁC NHẬN ỔN ĐỊNH: chạy lại GFlowNet %d seed với bộ tham số đã chọn "
        "(không sweep gì thêm ở bước này).", n_seeds_validation,
    )
    validation_result = multi_seed_validation(
        params, best_weights, best_beta_maxrules, best_training,
        valid_rules, cover, correct, rule_len, labels, device,
        n_seeds=n_seeds_validation, sweep_root=sweep_root,
    )
    summary["multi_seed_validation"] = {
        "summary_stats": validation_result["summary_stats"],
        "seeds": validation_result["seeds"],
        "detail_csv_path": validation_result["detail_csv_path"],
        "summary_csv_path": validation_result["summary_csv_path"],
        # Không lưu per_seed_records đầy đủ vào summary.json (đã có trong
        # detail_csv_path) để tránh file JSON phình to không cần thiết.
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        "Hoàn tất toàn bộ (sweep + xác nhận ổn định). Tóm tắt: %s", summary_path,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--n_trials_stage1", type=int, default=30,
                         help="Multi-objective (NSGA-II) cần nhiều trial hơn single-objective để "
                              "front hội tụ tốt — mặc định tăng từ 20 lên 30.")
    parser.add_argument("--n_trials_stage2", type=int, default=30,
                         help="Tương tự stage1 — tăng từ 15 lên 30.")
    parser.add_argument("--n_trials_stage3", type=int, default=6,
                         help="Giờ cũng multi-objective (NSGA-II) — mặc định giữ 6 vì mỗi "
                              "trial chạy full num_iterations (đắt), nhưng front sẽ khá thưa "
                              "với n_trials thấp. Tăng nếu đủ tài nguyên (khuyến nghị >=20).")
    parser.add_argument("--force_resweep", action="store_true",
                         help="Bỏ qua kết quả sweep cũ, chạy lại toàn bộ 3 giai đoạn từ đầu.")
    parser.add_argument("--pareto_lambda_conflict", type=float, default=0.3,
                         help="Trọng số CỐ ĐỊNH phạt redundancy_conflict khi chọn 1 điểm từ "
                              "Pareto front (chỉ dùng để chọn điểm, không dùng để lái search).")
    parser.add_argument("--n_seeds_validation", type=int, default=5,
                         help="Số seed chạy lại GFlowNet với bộ tham số đã chốt, để lấy mean±std "
                              "(khuyến nghị >=5).")
    parser.add_argument("--skip_multiseed_validation", action="store_true",
                         help="Bỏ qua hẳn bước xác nhận đa-seed (ví dụ khi chỉ muốn sweep tìm "
                              "tham số, chưa cần validate).")
    parser.add_argument("--only_multiseed_validation", action="store_true",
                         help="CHỈ chạy bước xác nhận đa-seed, dùng bộ tham số đã có sẵn trong "
                              "sweep_summary.json (bắt buộc phải tồn tại từ trước, KHÔNG sweep lại "
                              "dù có --force_resweep hay không). Tương đương gọi main() với "
                              "force_resweep=False và bỏ qua 3 giai đoạn sweep.")
    args = parser.parse_args()

    if args.only_multiseed_validation:
        _run_validation_only(
            args.config, n_seeds_validation=args.n_seeds_validation,
        )
    else:
        main(args.config, args.n_trials_stage1, args.n_trials_stage2, args.n_trials_stage3,
             force_resweep=args.force_resweep,
             pareto_lambda_conflict=args.pareto_lambda_conflict,
             n_seeds_validation=args.n_seeds_validation,
             skip_multiseed_validation=args.skip_multiseed_validation)
