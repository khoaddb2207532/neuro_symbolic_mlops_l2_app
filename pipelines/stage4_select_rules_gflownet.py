"""DVC Stage 4 — Dùng GFlowNet để chọn tập luật con tối ưu."""
import argparse
import os
import pickle

import torch

from src.gflownet.pipeline import ImprovedRuleExtractionPipelineV2
from src.rules.io import save_rules_excel
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main(params_path: str) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    features_dir = os.path.join(params["output_dir"], "02_features")
    rules_dir = os.path.join(params["output_dir"], "03_rules")
    output_dir = os.path.join(params["output_dir"], "04_filtered_rules")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(rules_dir, "valid_rules.pkl"), "rb") as f:
        valid_rules = pickle.load(f)

    train_features = torch.load(f"{features_dir}/train_features.pt").to(device)
    train_labels = torch.load(f"{features_dir}/train_labels.pt").to(device)
    val_features = torch.load(f"{features_dir}/val_features.pt").to(device)
    val_labels = torch.load(f"{features_dir}/val_labels.pt").to(device)

    gfn_cfg = params["gflownet"]
    pipeline = ImprovedRuleExtractionPipelineV2(
        device=device,
        proxy_cache_path=os.path.join(output_dir, "proxy_reward_cache.pth"),
        proxy_samples=gfn_cfg["proxy_samples"],
        proxy_epochs=gfn_cfg["proxy_epochs"],
    )
    selected_rules = pipeline.run(
        valid_rule_set=valid_rules,
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        max_rules=gfn_cfg["max_rules"],
        output_dir=output_dir,
        gfnet_hidden_dim=gfn_cfg["hidden_dim"],
        num_iterations=gfn_cfg["num_iterations"],
        batch_size=gfn_cfg["batch_size"],
        lr=gfn_cfg["lr"],
        logZ_lr=gfn_cfg["logZ_lr"],
        device=device,
        validation_interval=gfn_cfg["validation_interval"],
        loss_type=gfn_cfg["loss_type"],
        logZ_warmup_steps=gfn_cfg["logZ_warmup_steps"],
        val_samples=gfn_cfg["val_samples"],
    )

    with open(os.path.join(output_dir, "selected_rules_improved.pkl"), "wb") as f:
        pickle.dump(selected_rules, f)
    save_rules_excel(selected_rules, os.path.join(output_dir, "selected_rules_improved.xlsx"))
    logger.info("Stage 4 hoàn thành. %d luật được GFlowNet chọn.", len(selected_rules))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
