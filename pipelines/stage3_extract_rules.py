"""DVC Stage 3 — Huấn luyện Random Forest, trích luật thô, lọc bằng cross-validation."""
import argparse
import os
import pickle

import torch

from src.data.features import train_and_save_rf
from src.rules.extractor import RuleExtractor
from src.rules.io import save_rules_excel
from src.rules.validator import GPUFastRuleValidator
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed
import joblib

logger = get_logger(__name__)


def main(params_path: str) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    features_dir = os.path.join(params["output_dir"], "02_features")
    output_dir = os.path.join(params["output_dir"], "03_rules")
    os.makedirs(output_dir, exist_ok=True)

    rf_path = os.path.join(output_dir, "rf_model.joblib")
    train_and_save_rf(features_dir, rf_path)
    rf_model = joblib.load(rf_path)
    raw_rules = RuleExtractor().extract(rf_model)
    logger.info("Số luật thô trích từ RF: %d", len(raw_rules))

    train_features = torch.load(f"{features_dir}/train_features.pt").to(device)
    train_labels = torch.load(f"{features_dir}/train_labels.pt").to(device)

    validator = GPUFastRuleValidator(
        min_supp=params["rules"]["min_support"],
        min_conf=params["rules"]["min_confidence"],
    )
    valid_rules = validator.validate_crossval(train_features, train_labels, n_folds=params["rules"]["n_folds"])
    logger.info("Số luật hợp lệ sau cross-validation: %d", len(valid_rules.rules))

    with open(os.path.join(output_dir, "valid_rules_crossval.pkl"), "wb") as f:
        pickle.dump(valid_rules, f)

    save_rules_excel(valid_rules.rules, os.path.join(output_dir, "valid_rules_crossval.xlsx"))
    logger.info("Stage 3 hoàn thành. Kết quả tại %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
