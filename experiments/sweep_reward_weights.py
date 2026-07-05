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
import json
import os
import pickle
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
# Main — chạy tuần tự 3 giai đoạn
# --------------------------------------------------------------------------
def main(params_path: str, n_trials_stage1: int, n_trials_stage2: int, n_trials_stage3: int,
          force_resweep: bool = False,
          pareto_lambda_red: float = 0.3, pareto_lambda_comp: float = 0.2):
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sweep_root = os.path.join(params["output_dir"],"experiments")
    os.makedirs(sweep_root, exist_ok=True)
    storage_path = os.path.join(sweep_root, "optuna_study.db")
    summary_path = os.path.join(sweep_root, "sweep_summary.json")

    # --- Nếu đã có kết quả sweep trước đó và không ép chạy lại -> tái sử dụng ---
    if os.path.exists(summary_path) and not force_resweep:
        logger.info("Đã tìm thấy kết quả sweep trước đó tại %s — TÁI SỬ DỤNG, không chạy lại Optuna.", summary_path)
        with open(summary_path, "r") as f:
            summary = json.load(f)
        update_params_yaml(params_path, summary["best_weights"],
                            summary["best_beta_maxrules"], summary["best_training"])
        logger.info("Đã nạp lại bộ tham số cũ vào %s. Dùng --force_resweep nếu muốn sweep lại từ đầu.", params_path)
        return summary

    # --- Ngược lại: chạy sweep như bình thường ---
    valid_rules, cover, correct, rule_len = load_common_data(params, device)

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
    args = parser.parse_args()
    main(args.config, args.n_trials_stage1, args.n_trials_stage2, args.n_trials_stage3,
         force_resweep=args.force_resweep,
         pareto_lambda_red=args.pareto_lambda_red, pareto_lambda_comp=args.pareto_lambda_comp)