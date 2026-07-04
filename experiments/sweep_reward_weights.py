"""Script thử nghiệm hyperparameter cho reward — KHÔNG phải DVC stage,
chạy độc lập để tìm bộ tham số tốt trước khi đưa vào params.yaml."""
import json
import os
import random
import torch

from src.gflownet.pipeline import RuleExtractionPipeline
from src.gflownet.evaluation import evaluate_run
from src.rules.validator import RuleValidator
from src.utils.config import load_params
from src.utils.seed import set_seed
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_common_data(params_path: str, device: str):
    """Tái sử dụng đúng logic load dữ liệu như stage4, tránh lặp code
    load/lọc rule mỗi lần sweep (chỉ làm 1 lần, dùng chung cho mọi trial)."""
    params = load_params(params_path)
    features_dir = os.path.join(params["output_dir"], "02_features")
    rules_dir = os.path.join(params["output_dir"], "03_rules")

    import pickle
    with open(os.path.join(rules_dir, "raw_rules.pkl"), "rb") as f:
        raw_rules = pickle.load(f)

    train_features = torch.load(f"{features_dir}/train_features.pt").to(device)
    train_labels = torch.load(f"{features_dir}/train_labels.pt").to(device)
    val_features = torch.load(f"{features_dir}/val_features.pt").to(device)
    val_labels = torch.load(f"{features_dir}/val_labels.pt").to(device)

    validator = RuleValidator(
        min_supp=params["rules"]["min_support"],
        min_conf=params["rules"]["min_confidence"],
    )
    valid_rule_set, cover, correct, rule_len = validator.validate_and_build_tensors(
        raw_rules, val_features, val_labels, store_device=device
    )
    return list(valid_rule_set.rules), cover, correct, rule_len, params


def run_sweep(params_path: str = "params.yaml", n_trials: int = 20, seed: int = 42):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    valid_rules, cover, correct, rule_len, params = load_common_data(params_path, device)
    max_rules = params["gflownet"]["max_rules"]

    configs = [
        {
            "w_acc": round(random.uniform(0.7, 1.3), 2),
            "w_cov": round(random.uniform(0.2, 0.8), 2),
            "w_red": round(random.uniform(0.1, 0.5), 2),
            "w_comp": round(random.uniform(0.05, 0.35), 2),
            "beta": 3.0,  # cố định ở giai đoạn 1, sweep riêng ở giai đoạn 2
        }
        for _ in range(n_trials)
    ]

    results = []
    for i, cfg in enumerate(configs):
        logger.info("=== Trial %d/%d: %s ===", i + 1, n_trials, cfg)
        output_dir = f"/tmp/sweep_A/trial_{i}"
        os.makedirs(output_dir, exist_ok=True)

        pipeline = RuleExtractionPipeline(device=device, **cfg)
        selected = pipeline.run(
            valid_rules=valid_rules, cover=cover, correct=correct, rule_len=rule_len,
            max_rules=max_rules, output_dir=output_dir,
            gfnet_hidden_dim=params["gflownet"]["hidden_dim"],
            num_iterations=800,  # ngắn hơn full run để sweep nhanh
            batch_size=params["gflownet"]["batch_size"],
            lr=params["gflownet"]["lr"], logZ_lr=params["gflownet"]["logZ_lr"],
            device=device, validation_interval=100, loss_type="tb",
            logZ_warmup_steps=50, val_samples=10,
        )

        reward_module = getattr(pipeline, "_last_reward_module", None)
        metric = evaluate_run(selected, valid_rules, reward_module)
        metric.update(cfg)
        metric["trial"] = i
        results.append(metric)
        logger.info("Result: %s", metric)

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/sweep_reward_weights.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    results.sort(key=lambda r: r["f1_like"], reverse=True)
    logger.info("Top 5 configs:")
    for r in results[:5]:
        logger.info(r)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()
    run_sweep(args.config, args.n_trials)