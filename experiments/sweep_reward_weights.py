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
def load_common_data(params: dict, device: str) -> Tuple[List[Rule], torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return valid_rules, cover, correct, rule_len


# --------------------------------------------------------------------------
# Hàm chạy 1 lần train+eval, dùng chung cho cả 3 giai đoạn
# --------------------------------------------------------------------------
def run_one_trial(
    cfg: Dict,
    valid_rules: List[Rule],
    cover: torch.Tensor,
    correct: torch.Tensor,
    rule_len: torch.Tensor,
    max_rules: int,
    num_iterations: int,
    device: str,
    output_dir: str,
    fixed_train_kwargs: Dict,
) -> Dict:
    pipeline = RuleExtractionPipeline(
        device=device,
        w_acc=cfg["w_acc"], w_cov=cfg["w_cov"],
        w_red=cfg["w_red"], w_comp=cfg["w_comp"],
        beta=cfg["beta"],
    )
    selected = pipeline.run(
        valid_rules=valid_rules, cover=cover, correct=correct, rule_len=rule_len,
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
#     4 chiều (accuracy, coverage, redundancy, complexity), không thiên vị.
#   - Sau khi có Pareto front (nhiều lựa chọn không ai thống trị ai), NGƯỜI
#     NGHIÊN CỨU mới áp "khẩu vị" của mình (lambda_red, lambda_comp) để chọn
#     RA 1 điểm — vì pipeline sau (stage2, stage3, ghi params.yaml) cần 1 bộ
#     trọng số duy nhất, không thể mang cả 1 tập nghiệm đi tiếp.
#   - lambda_red/lambda_comp ở đây là hằng số CỐ ĐỊNH do người dùng chọn,
#     hoàn toàn tách biệt khỏi w_red/w_comp (là biến được GFlowNet sweep) —
#     không có chuyện tự thưởng cho chính tham số đang được tối ưu.
# --------------------------------------------------------------------------
def select_from_pareto_front(
    study: "optuna.Study",
    lambda_red: float = 0.3,
    lambda_comp: float = 0.2,
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
        accuracy, coverage, redundancy, complexity = t.values
        return accuracy + coverage - lambda_red * redundancy - lambda_comp * complexity

    chosen = max(pareto_trials, key=scalarize)
    logger.info(
        "[%s] Pareto front: %d điểm. Chọn trial=%d (lambda_red=%.2f, lambda_comp=%.2f) "
        "-> (accuracy,coverage,redundancy,complexity)=%s, params=%s",
        study.study_name, len(pareto_trials), chosen.number,
        lambda_red, lambda_comp, chosen.values, chosen.params,
    )
    return chosen


# --------------------------------------------------------------------------
# GIAI ĐOẠN 1 — sweep w_acc / w_cov / w_red / w_comp (reward shape)
# beta và max_rules giữ cố định ở giá trị mặc định từ params.yaml
#
# ĐÃ CHUYỂN SANG MULTI-OBJECTIVE (NSGA-II): objective trả về TUYỆT ĐỐI 4 chiều
# (accuracy, coverage, redundancy, complexity) thay vì scalar f1_like — vì
# f1_like không chứa redundancy/complexity nên trước đây w_red/w_comp bị đẩy
# về cận dưới của khoảng sweep một cách có hệ thống (không có tín hiệu phản
# hồi). Xem thảo luận trước khi đổi.
# --------------------------------------------------------------------------
def stage1_reward_weights(
    params: dict, valid_rules, cover, correct, rule_len, device: str,
    n_trials: int, storage_path: str, sweep_root: str,
    pareto_lambda_red: float = 0.3, pareto_lambda_comp: float = 0.2,
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
            "w_red": trial.suggest_float("w_red", 0.1, 0.5),
            "w_comp": trial.suggest_float("w_comp", 0.05, 0.35),
            "beta": 3.0,  # cố định ở giai đoạn 1
        }
        set_seed(params["seed"] + trial.number)
        output_dir = os.path.join(sweep_root, "stage1", f"trial_{trial.number}")
        os.makedirs(output_dir, exist_ok=True)

        metric = run_one_trial(cfg, valid_rules, cover, correct, rule_len,
                                max_rules, num_iterations, device, output_dir, fixed_kwargs)
        for k, v in metric.items():
            trial.set_user_attr(k, v)
        logger.info("[Stage1] Trial %d: %s", trial.number, metric)
        # 4 mục tiêu riêng biệt — KHÔNG gộp thành 1 scalar ở đây (xem docstring
        # select_from_pareto_front phía trên vì sao).
        return metric["accuracy"], metric["coverage"], metric["redundancy"], metric["complexity"]

    study = optuna.create_study(
        study_name="stage1_reward_weights",
        directions=["maximize", "maximize", "minimize", "minimize"],
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(seed=params["seed"]),
    )
    remaining = n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    else:
        logger.info("[Stage1] Đã đủ %d trials từ lần chạy trước, bỏ qua.", n_trials)

    chosen = select_from_pareto_front(study, pareto_lambda_red, pareto_lambda_comp)
    return {**chosen.params, "beta": 3.0}  # trả về w_acc/w_cov/w_red/w_comp đã chọn từ Pareto front


# --------------------------------------------------------------------------
# GIAI ĐOẠN 2 — sweep beta + max_rules, dùng w_* tốt nhất từ giai đoạn 1
# --------------------------------------------------------------------------
def stage2_beta_maxrules(
    params: dict, best_weights: Dict, valid_rules, cover, correct, rule_len, device: str,
    n_trials: int, storage_path: str, sweep_root: str,
    pareto_lambda_red: float = 0.3, pareto_lambda_comp: float = 0.2,
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
            "w_red": best_weights["w_red"], "w_comp": best_weights["w_comp"],
            "beta": trial.suggest_float("beta", 1.0, 8.0),
        }
        max_rules = trial.suggest_categorical("max_rules", [16, 24, 32, 40, 48])
        set_seed(params["seed"] + trial.number)
        output_dir = os.path.join(sweep_root, "stage2", f"trial_{trial.number}")
        os.makedirs(output_dir, exist_ok=True)

        metric = run_one_trial(cfg, valid_rules, cover, correct, rule_len,
                                max_rules, num_iterations, device, output_dir, fixed_kwargs)
        for k, v in metric.items():
            trial.set_user_attr(k, v)
        logger.info("[Stage2] Trial %d: %s", trial.number, metric)
        # Cùng lý do như stage1: max_rules cũng là 1 "biến đang được sweep",
        # nếu chấm điểm bằng f1_like (không phạt complexity) nó sẽ bị đẩy lên
        # cận trên (48) một cách có hệ thống, vô hiệu hoá mục đích chọn budget
        # gọn. Trả 4 chiều riêng để NSGA-II khám phá đúng đánh đổi.
        return metric["accuracy"], metric["coverage"], metric["redundancy"], metric["complexity"]

    study = optuna.create_study(
        study_name="stage2_beta_maxrules",
        directions=["maximize", "maximize", "minimize", "minimize"],
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(seed=params["seed"]),
    )
    remaining = n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    else:
        logger.info("[Stage2] Đã đủ %d trials từ lần chạy trước, bỏ qua.", n_trials)

    chosen = select_from_pareto_front(study, pareto_lambda_red, pareto_lambda_comp)
    return {"beta": chosen.params["beta"], "max_rules": chosen.params["max_rules"]}


# --------------------------------------------------------------------------
# GIAI ĐOẠN 3 — sweep lr/batch_size, dùng reward + beta + max_rules tốt nhất
# từ giai đoạn 1 và 2. Chạy full num_iterations vì mục đích là xác nhận
# hội tụ, không phải dò nhanh như 2 giai đoạn trước.
#
# ĐÃ CHUYỂN SANG MULTI-OBJECTIVE cho NHẤT QUÁN với stage1/2 — dù lr/batch_size
# không trực tiếp điều khiển redundancy/complexity như w_red/w_comp hay
# max_rules, nhưng động lực học (lr, batch_size) vẫn có thể ảnh hưởng tới
# CHẤT LƯỢNG hội tụ trên cả 4 chiều (ví dụ lr quá lớn có thể hội tụ về 1 tập
# luật kém đa dạng hơn dù cùng reward weights) — dùng cùng 1 cơ chế đánh giá
# xuyên suốt 3 giai đoạn tránh việc so sánh "táo với cam" giữa các giai đoạn.
# --------------------------------------------------------------------------
def stage3_training_dynamics(
    params: dict, best_weights: Dict, best_beta_maxrules: Dict,
    valid_rules, cover, correct, rule_len, device: str,
    n_trials: int, storage_path: str, sweep_root: str,
    pareto_lambda_red: float = 0.3, pareto_lambda_comp: float = 0.2,
) -> Dict:
    gfn_cfg = params["gflownet"]
    max_rules = best_beta_maxrules["max_rules"]
    num_iterations = gfn_cfg["num_iterations"]  # full, không rút gọn ở giai đoạn cuối

    def objective(trial: optuna.Trial):
        cfg = {
            "w_acc": best_weights["w_acc"], "w_cov": best_weights["w_cov"],
            "w_red": best_weights["w_red"], "w_comp": best_weights["w_comp"],
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

        metric = run_one_trial(cfg, valid_rules, cover, correct, rule_len,
                                max_rules, num_iterations, device, output_dir, fixed_kwargs)
        metric["lr"] = lr
        metric["batch_size"] = batch_size
        for k, v in metric.items():
            trial.set_user_attr(k, v)
        logger.info("[Stage3] Trial %d: %s", trial.number, metric)
        return metric["accuracy"], metric["coverage"], metric["redundancy"], metric["complexity"]

    study = optuna.create_study(
        study_name="stage3_training_dynamics",
        directions=["maximize", "maximize", "minimize", "minimize"],
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(seed=params["seed"]),
    )
    remaining = n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    else:
        logger.info("[Stage3] Đã đủ %d trials từ lần chạy trước, bỏ qua.", n_trials)

    chosen = select_from_pareto_front(study, pareto_lambda_red, pareto_lambda_comp)
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
    valid_rules, cover, correct, rule_len,
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
        "w_red": best_weights["w_red"], "w_comp": best_weights["w_comp"],
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

    metric_keys = ["n_rules", "accuracy", "coverage", "redundancy", "complexity", "f1_like"]
    records = []
    for i, seed in enumerate(seeds):
        set_seed(seed)
        output_dir = os.path.join(sweep_root, "multi_seed_validation", f"seed_{seed}")
        os.makedirs(output_dir, exist_ok=True)
        metric = run_one_trial(cfg, valid_rules, cover, correct, rule_len,
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
    # Cột "complexity"/"complexity_std" và "reward_score/reward_value" không
    # có sẵn ở CSV cũ (định nghĩa nằm ở script khác) — bạn tự đối chiếu thêm
    # nếu cần, không tự suy đoán công thức reward_score/reward_value ở đây.
    summary_csv_path = os.path.join(sweep_root, "gflownet_multiseed_summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method", "budget", "budget_value",
            "so_luat", "so_luat_std",
            "accuracy", "accuracy_std",
            "coverage", "coverage_std",
            "redundancy", "redundancy_std",
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
    gfn_cfg["w_red"] = best_weights["w_red"]
    gfn_cfg["w_comp"] = best_weights["w_comp"]
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

    valid_rules, cover, correct, rule_len = load_common_data(params, device)

    validation_result = multi_seed_validation(
        params, best_weights, best_beta_maxrules, best_training,
        valid_rules, cover, correct, rule_len, device,
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
          pareto_lambda_red: float = 0.3, pareto_lambda_comp: float = 0.2,
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
    valid_rules, cover, correct, rule_len = load_common_data(params, device)

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
            params, valid_rules, cover, correct, rule_len, device,
            n_trials=n_trials_stage1, storage_path=storage_path, sweep_root=sweep_root,
            pareto_lambda_red=pareto_lambda_red, pareto_lambda_comp=pareto_lambda_comp,
        )

        logger.info("GIAI ĐOẠN 2: Sweep beta + max_rules (multi-objective)")
        best_beta_maxrules = stage2_beta_maxrules(
            params, best_weights, valid_rules, cover, correct, rule_len, device,
            n_trials=n_trials_stage2, storage_path=storage_path, sweep_root=sweep_root,
            pareto_lambda_red=pareto_lambda_red, pareto_lambda_comp=pareto_lambda_comp,
        )

        logger.info("GIAI ĐOẠN 3: Sweep lr/batch_size (multi-objective)")
        best_training = stage3_training_dynamics(
            params, best_weights, best_beta_maxrules, valid_rules, cover, correct, rule_len, device,
            n_trials=n_trials_stage3, storage_path=storage_path, sweep_root=sweep_root,
            pareto_lambda_red=pareto_lambda_red, pareto_lambda_comp=pareto_lambda_comp,
        )

        update_params_yaml(params_path, best_weights, best_beta_maxrules, best_training)

        summary = {
            "best_weights": best_weights,
            "best_beta_maxrules": best_beta_maxrules,
            "best_training": best_training,
            # Ghi lại để tái lập đúng lựa chọn (đây là "khẩu vị" cố định dùng để
            # chọn 1 điểm trên Pareto front, KHÔNG phải tham số được sweep).
            "pareto_selection": {
                "lambda_red": pareto_lambda_red,
                "lambda_comp": pareto_lambda_comp,
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
        valid_rules, cover, correct, rule_len, device,
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
    parser.add_argument("--pareto_lambda_red", type=float, default=0.3,
                         help="Trọng số CỐ ĐỊNH phạt redundancy khi chọn 1 điểm từ Pareto front "
                              "(chỉ dùng để chọn điểm, không dùng để lái search).")
    parser.add_argument("--pareto_lambda_comp", type=float, default=0.2,
                         help="Trọng số CỐ ĐỊNH phạt complexity khi chọn 1 điểm từ Pareto front.")
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
             pareto_lambda_red=args.pareto_lambda_red, pareto_lambda_comp=args.pareto_lambda_comp,
             n_seeds_validation=args.n_seeds_validation,
             skip_multiseed_validation=args.skip_multiseed_validation)