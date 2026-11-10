"""Pipeline chạy liên hoàn Baseline CNN + CBM + Rule Mining + GFlowNet + Rule-Regularized CBM.

Thứ tự thực thi đầy đủ (có Baseline đối chứng):
  1. Stage 1:  Huấn luyện Baseline CNN (đối chứng chuẩn, outputs/01_baseline/).
  2. Stage 1b: Huấn luyện Concept Bottleneck Model (CBM) với Concept Loss (outputs/cbm/).
  3. Stage 2:  Trích xuất Concept Probabilities (P(c) ∈ [0, 1]) từ CBM (outputs/02_features/).
  4. Stage 3:  Trích xuất luật ngữ nghĩa (Human-interpretable rules) bằng Random Forest (outputs/03_rules/).
  5. Stage 4:  GFlowNet tuyển chọn tập luật Concept tối ưu và đa dạng (outputs/04_filtered_rules/).
  6. Stage 5:  Huấn luyện CBM có ràng buộc luật (Rule-Regularized Fine-Tuning, outputs/05_rules_model/).

Chạy:
  python -m pipelines.run_cbm_neuro_symbolic_pipeline --config params.yaml
"""
import argparse
import sys
import time

from pipelines import (
    stage1_train_baseline,
    stage1b_train_cbm,
    stage2_extract_features,
    stage3_extract_rules,
    stage4_select_rules_gflownet,
    stage5_train_rule_regularized,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


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
