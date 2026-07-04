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
# GIAI ĐOẠN 1 — sweep w_acc / w_cov / w_red / w_comp (reward shape)
# beta và max_rules giữ cố định ở giá trị mặc định từ params.yaml
# --------------------------------------------------------------------------
def stage1_reward_weights(
    params: dict, valid_rules, cover, correct, rule_len, device: str,
    n_trials: int, storage_path: str, sweep_root: str,
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

    def objective(trial: optuna.Trial) -> float:
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
        return metric["f1_like"]

    study = optuna.create_study(
        study_name="stage1_reward_weights",
        direction="maximize",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )
    remaining = n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    else:
        logger.info("[Stage1] Đã đủ %d trials từ lần chạy trước, bỏ qua.", n_trials)

    best = study.best_trial
    logger.info("[Stage1] BEST: value=%.4f, params=%s", best.value, best.params)
    return {**best.params, "beta": 3.0}  # trả về w_acc/w_cov/w_red/w_comp tốt nhất


# --------------------------------------------------------------------------
# GIAI ĐOẠN 2 — sweep beta + max_rules, dùng w_* tốt nhất từ giai đoạn 1
# --------------------------------------------------------------------------
def stage2_beta_maxrules(
    params: dict, best_weights: Dict, valid_rules, cover, correct, rule_len, device: str,
    n_trials: int, storage_path: str, sweep_root: str,
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

    def objective(trial: optuna.Trial) -> float:
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
        return metric["f1_like"]

    study = optuna.create_study(
        study_name="stage2_beta_maxrules",
        direction="maximize",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )
    remaining = n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    else:
        logger.info("[Stage2] Đã đủ %d trials từ lần chạy trước, bỏ qua.", n_trials)

    best = study.best_trial
    logger.info("[Stage2] BEST: value=%.4f, params=%s", best.value, best.params)
    return {"beta": best.params["beta"], "max_rules": best.params["max_rules"]}


# --------------------------------------------------------------------------
# GIAI ĐOẠN 3 — sweep lr/batch_size, dùng reward + beta + max_rules tốt nhất
# từ giai đoạn 1 và 2. Chạy full num_iterations vì mục đích là xác nhận
# hội tụ, không phải dò nhanh như 2 giai đoạn trước.
# --------------------------------------------------------------------------
def stage3_training_dynamics(
    params: dict, best_weights: Dict, best_beta_maxrules: Dict,
    valid_rules, cover, correct, rule_len, device: str,
    n_trials: int, storage_path: str, sweep_root: str,
) -> Dict:
    gfn_cfg = params["gflownet"]
    max_rules = best_beta_maxrules["max_rules"]
    num_iterations = gfn_cfg["num_iterations"]  # full, không rút gọn ở giai đoạn cuối

    def objective(trial: optuna.Trial) -> float:
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
        return metric["f1_like"]

    study = optuna.create_study(
        study_name="stage3_training_dynamics",
        direction="maximize",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )
    remaining = n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    else:
        logger.info("[Stage3] Đã đủ %d trials từ lần chạy trước, bỏ qua.", n_trials)

    best = study.best_trial
    logger.info("[Stage3] BEST: value=%.4f, params=%s", best.value, best.params)
    return {"lr": best.params["lr"], "batch_size": best.params["batch_size"]}


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
          force_resweep: bool = False):
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

    logger.info("GIAI ĐOẠN 1: Sweep reward weights")
    best_weights = stage1_reward_weights(
        params, valid_rules, cover, correct, rule_len, device,
        n_trials=n_trials_stage1, storage_path=storage_path, sweep_root=sweep_root,
    )

    logger.info("GIAI ĐOẠN 2: Sweep beta + max_rules")
    best_beta_maxrules = stage2_beta_maxrules(
        params, best_weights, valid_rules, cover, correct, rule_len, device,
        n_trials=n_trials_stage2, storage_path=storage_path, sweep_root=sweep_root,
    )

    logger.info("GIAI ĐOẠN 3: Sweep lr/batch_size")
    best_training = stage3_training_dynamics(
        params, best_weights, best_beta_maxrules, valid_rules, cover, correct, rule_len, device,
        n_trials=n_trials_stage3, storage_path=storage_path, sweep_root=sweep_root,
    )

    update_params_yaml(params_path, best_weights, best_beta_maxrules, best_training)

    summary = {
        "best_weights": best_weights,
        "best_beta_maxrules": best_beta_maxrules,
        "best_training": best_training,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Sweep hoàn tất, kết quả lưu tại %s để tái sử dụng về sau.", summary_path)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--n_trials_stage1", type=int, default=20)
    parser.add_argument("--n_trials_stage2", type=int, default=15)
    parser.add_argument("--n_trials_stage3", type=int, default=6)
    parser.add_argument("--force_resweep", action="store_true",
                         help="Bỏ qua kết quả sweep cũ, chạy lại toàn bộ 3 giai đoạn từ đầu.")
    args = parser.parse_args()
    main(args.config, args.n_trials_stage1, args.n_trials_stage2, args.n_trials_stage3,
         force_resweep=args.force_resweep)