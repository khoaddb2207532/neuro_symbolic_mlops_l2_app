"""DVC Stage 3 — Huấn luyện Random Forest (1 lần duy nhất), trích luật thô,
lọc luật bằng val set thật (không còn train lại RF trong bước lọc — xem
src/rules/validator.py và src/data/features.py để biết lý do đã gộp)."""
import argparse
import os
import pickle

import torch

from src.data.features import train_and_save_rf
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
    output_dir = os.path.join(params["output_dir"], "03_rules")
    os.makedirs(output_dir, exist_ok=True)

    # Train RF 1 lần + trích luật thô ngay trong hàm này (nơi DUY NHẤT làm
    # việc này trong toàn bộ pipeline).
    raw_rules = train_and_save_rf(
        features_dir=features_dir,
        rf_output_path=os.path.join(output_dir, "rf_model.joblib"),
        rules_output_path=os.path.join(output_dir, "raw_rules.pkl"),
        n_estimators=params["rf"]["n_estimators"],
        max_depth=params["rf"]["max_depth"],
    )
    logger.info("Số luật thô trích từ RF: %d", len(raw_rules))

    raw_rules_path = os.path.join(output_dir, "raw_rules.pkl")
    with open(raw_rules_path, "wb") as f:
        pickle.dump(raw_rules, f)

    save_rules_excel(raw_rules.rules, os.path.join(output_dir, "raw_rules.xlsx"))
    
    logger.info("Stage 3 hoàn thành. Kết quả tại %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
