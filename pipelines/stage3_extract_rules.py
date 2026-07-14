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

import json
import numpy as np
import torch
from src.evaluation.rf_eval import evaluate_rf, save_rf_evaluation

logger = get_logger(__name__)


def main(params_path: str) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    features_dir = os.path.join(params["output_dir"], "02_features")
    output_dir = os.path.join(params["output_dir"], "03_rules")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Train RF và trích luật (giữ nguyên) ----
    raw_rules = train_and_save_rf(
        features_dir=features_dir,
        rf_output_path=os.path.join(output_dir, "rf_model.joblib"),
        rules_output_path=os.path.join(output_dir, "raw_rules.pkl"),
        n_estimators=params["rf"]["n_estimators"],
        max_depth=params["rf"]["max_depth"],
    )
    logger.info("Số luật thô trích từ RF: %d", len(raw_rules))

    # Lưu luật thô
    raw_rules_path = os.path.join(output_dir, "raw_rules.pkl")
    with open(raw_rules_path, "wb") as f:
        pickle.dump(raw_rules, f)
    save_rules_excel(raw_rules.rules, os.path.join(output_dir, "raw_rules.xlsx"))

    # ---- BỔ SUNG: ĐÁNH GIÁ RF SO VỚI CNN ----
    # Load model RF vừa train (nếu cần, nhưng đã có rf_model trong train_and_save_rf?
    # Hiện tại train_and_save_rf trả về raw_rules, KHÔNG trả model.
    # Ta sẽ load lại model từ file đã lưu.
    import joblib
    rf_model = joblib.load(os.path.join(output_dir, "rf_model.joblib"))

    # Load features và logits của validation và test
    splits = ['val', 'test']  # test có thể có hoặc không, nhưng nên đánh giá
    for split in splits:
        feat_path = os.path.join(features_dir, f'{split}_features.npy')
        label_path = os.path.join(features_dir, f'{split}_labels.npy')
        logits_path = os.path.join(features_dir, f'{split}_logits.pt')
        if not (os.path.exists(feat_path) and os.path.exists(label_path) and os.path.exists(logits_path)):
            logger.warning(f"Bỏ qua {split} vì thiếu file")
            continue

        X = np.load(feat_path)
        y = np.load(label_path)
        logits = torch.load(logits_path).numpy()  # shape (n, num_classes)

        eval_result = evaluate_rf(rf_model, X, y, cnn_logits=logits)
        # Lưu riêng cho từng split
        save_path = os.path.join(output_dir, f'rf_evaluation_{split}.json')
        save_rf_evaluation(eval_result, save_path)

        # In log các chỉ số chính
        logger.info(f"===== RF evaluation on {split} =====")
        logger.info(f"RF accuracy: {eval_result['accuracy']:.4f}")
        logger.info(f"CNN accuracy: {eval_result['cnn_accuracy']:.4f}")
        logger.info(f"Disagreement rate: {eval_result['disagreement_rate']:.4f}")
        logger.info(f"RF avg entropy: {eval_result['avg_entropy_rf']:.4f}")
        logger.info(f"CNN avg entropy: {eval_result['cnn_avg_entropy']:.4f}")
        logger.info(f"rf_correct_cnn_wrong: {eval_result['rf_correct_cnn_wrong']:.4f}")
        logger.info(f"cnn_correct_rf_wrong: {eval_result['cnn_correct_rf_wrong']:.4f}")

    logger.info("Stage 3 hoàn thành. Kết quả tại %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
