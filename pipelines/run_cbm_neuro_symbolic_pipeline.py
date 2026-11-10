"""Pipeline chạy liên hoàn Baseline CNN + CBM + Rule Mining + GFlowNet + Rule-Regularized CBM.

Thứ tự thực thi đầy đủ (có Baseline đối chứng):
  1. Stage 1:  Huấn luyện Baseline CNN (đối chứng chuẩn, outputs/01_baseline/).
  2. Stage 1b: Huấn luyện Concept Bottleneck Model (CBM) với Concept Loss (outputs/cbm/).
  3. Stage 2:  Trích xuất Concept Probabilities (P(c) ∈ [0, 1]) từ CBM (outputs/02_features/).
  4. Stage 3:  Trích xuất luật ngữ nghĩa (Human-interpretable rules) bằng Random Forest (outputs/03_rules/).
  5. Stage 4:  GFlowNet tuyển chọn tập luật Concept tối ưu và đa dạng (outputs/04_filtered_rules/).
  6. Stage 5:  Huấn luyện CBM có ràng buộc luật (Rule-Regularized Fine-Tuning, outputs/05_rules_model/).
  7. Lưu tổng hợp Classification Report cho mọi mô hình trong seed.

Chạy:
  python -m pipelines.run_cbm_neuro_symbolic_pipeline --config params.yaml
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from pipelines import (
    stage1_train_baseline,
    stage1b_train_cbm,
    stage2_extract_features,
    stage3_extract_rules,
    stage4_select_rules_gflownet,
    stage5_train_rule_regularized,
)
from src.utils.config import load_params
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def generate_seed_classification_summary(config_path: str) -> None:
    """Tổng hợp chi tiết classification report của từng mô hình trong seed."""
    try:
        import pandas as pd
        params = load_params(config_path)
        output_dir = Path(params.get("output_dir", "outputs"))
        seed = params.get("seed", 42)

        models_info = [
            ("Baseline CNN (Hộp đen)", output_dir / "01_baseline"),
            ("CBM (Concept Bottleneck)", output_dir / "cbm"),
            ("CBM + Rule-Regularized", output_dir / "05_rules_model"),
        ]

        rows = []
        for model_name, path in models_info:
            json_files = list(path.glob("*classification_report.json"))
            if not json_files:
                continue
            with open(json_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)

            row = {
                "Seed": seed,
                "Model": model_name,
                "Accuracy (%)": round(data.get("accuracy", 0.0) * 100, 2),
                "Macro F1 (%)": round(data.get("macro avg", {}).get("f1-score", 0.0) * 100, 2),
                "Weighted F1 (%)": round(data.get("weighted avg", {}).get("f1-score", 0.0) * 100, 2),
            }
            # Per-class F1
            for k, v in data.items():
                if k not in ("accuracy", "macro avg", "weighted avg") and isinstance(v, dict):
                    row[f"F1_{k}"] = round(v.get("f1-score", 0.0) * 100, 2)
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            summary_csv = output_dir / "seed_classification_summary.csv"
            df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
            logger.info("Đã lưu bảng tổng hợp Classification Report của Seed %d tại: %s", seed, summary_csv)
    except Exception as e:
        logger.warning("Không thể tổng hợp classification summary: %s", e)


def main(
    config_path: str,
    skip_baseline: bool = False,
    skip_cbm: bool = False,
) -> None:
    total_start = time.time()
    logger.info("=" * 70)
    logger.info("BẮT ĐẦU PIPELINE TÍCH HỢP BASELINE + CBM + GFLOWNET + RULE REGULARIZATION")
    logger.info("=" * 70)

    # 1. Stage 1: Train Baseline CNN (Mô hình đối chứng)
    if not skip_baseline:
        logger.info("\n>>> [1/6] STAGE 1: Huấn luyện Baseline CNN (mô hình đối chứng)...")
        stage1_train_baseline.main(config_path)
    else:
        logger.info("\n>>> [1/6] BỎ QUA STAGE 1 (sử dụng checkpoint Baseline có sẵn).")

    # 2. Stage 1b: Train CBM
    if not skip_cbm:
        logger.info("\n>>> [2/6] STAGE 1b: Huấn luyện CBM với Concept Loss...")
        stage1b_train_cbm.main(config_path)
    else:
        logger.info("\n>>> [2/6] BỎ QUA STAGE 1b (sử dụng checkpoint CBM có sẵn).")

    # 3. Stage 2: Extract Concept Features
    logger.info("\n>>> [3/6] STAGE 2: Trích Concept Probabilities từ CBM...")
    stage2_extract_features.main(config_path, source_override="cbm")

    # 4. Stage 3: Extract Human-interpretable Rules
    logger.info("\n>>> [4/6] STAGE 3: Trích luật ngữ nghĩa (IF Concept THEN Class)...")
    stage3_extract_rules.main(config_path)

    # 5. Stage 4: GFlowNet Rule Selection
    logger.info("\n>>> [5/6] STAGE 4: GFlowNet tuyển chọn tập luật Concept...")
    stage4_select_rules_gflownet.main(config_path)

    # 6. Stage 5: Rule-Regularized Training on CBM
    logger.info("\n>>> [6/6] STAGE 5: Fine-tune CBM với Rule Penalty...")
    stage5_train_rule_regularized.main(config_path, target_override="cbm")

    # 7. Tổng hợp bảng Classification Report
    generate_seed_classification_summary(config_path)

    elapsed = (time.time() - total_start) / 60
    logger.info("\n" + "=" * 70)
    logger.info("HOÀN THÀNH TOÀN BỘ PIPELINE TÍCH HỢP TRONG %.2f PHÚT!", elapsed)
    logger.info("  1. Baseline CNN checkpoint:  outputs/01_baseline/baseline_best.pth")
    logger.info("  2. CBM Model checkpoint:     outputs/cbm/cbm_best.pth")
    logger.info("  3. GFlowNet Rules Excel:     outputs/04_filtered_rules/selected_rules.xlsx")
    logger.info("  4. Rule-Regularized CBM:     outputs/05_rules_model/rule_regularized_best.pth")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Bỏ qua huấn luyện Baseline CNN nếu đã có sẵn",
    )
    parser.add_argument(
        "--skip-cbm",
        action="store_true",
        help="Bỏ qua huấn luyện CBM nếu đã có sẵn",
    )
    args = parser.parse_args()
    main(args.config, skip_baseline=args.skip_baseline, skip_cbm=args.skip_cbm)
