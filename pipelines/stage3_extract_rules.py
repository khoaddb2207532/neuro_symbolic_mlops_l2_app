"""DVC Stage 3 — Huấn luyện Random Forest (1 lần duy nhất), trích luật thô,
lọc luật bằng val set thật (không còn train lại RF trong bước lọc — xem
src/rules/validator.py và src/data/features.py để biết lý do đã gộp)."""
import argparse
import os
import pickle

import torch

from src.data.features import train_and_save_rf
from src.rules.io import save_rules_excel
from src.rules.validator import GPUFastRuleValidator
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
    )

    # Lọc luật bằng VAL SET THẬT (ảnh CNN chưa từng thấy khi train) — không
    # dùng lại train_features/K-Fold nữa (xem lý do trong validator.py).
    val_features = torch.load(f"{features_dir}/val_features.pt").to(device)
    val_labels = torch.load(f"{features_dir}/val_labels.pt").to(device)

    validator = GPUFastRuleValidator(
        min_supp=params["rules"]["min_support"],
        min_conf=params["rules"]["min_confidence"],
    )
    valid_rules = validator.validate(raw_rules, val_features, val_labels)
    logger.info("Số luật hợp lệ sau khi lọc bằng val set: %d", len(valid_rules))

    valid_rules_path = os.path.join(output_dir, "valid_rules.pkl")
    with open(valid_rules_path, "wb") as f:
        pickle.dump(valid_rules, f)

    save_rules_excel(valid_rules.rules, os.path.join(output_dir, "valid_rules.xlsx"))
    logger.info("Stage 3 hoàn thành. Kết quả tại %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
