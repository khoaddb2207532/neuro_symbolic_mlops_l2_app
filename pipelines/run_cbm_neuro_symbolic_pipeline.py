"""Pipeline chạy liên hoàn CBM + Rule Mining + GFlowNet + Rule-Regularized CBM.

Thứ tự thực thi:
  1. Stage 1b: Huấn luyện Concept Bottleneck Model (CBM) với Concept Loss.
  2. Stage 2:  Trích xuất Concept Probabilities (P(c) ∈ [0, 1]) từ CBM đã huấn luyện.
  3. Stage 3:  Trích xuất luật ngữ nghĩa (Human-interpretable rules) bằng Random Forest.
  4. Stage 4:  GFlowNet tuyển chọn tập luật Concept tối ưu và đa dạng.
  5. Stage 5:  Huấn luyện CBM có ràng buộc luật (Rule-Regularized Fine-Tuning).

Chạy:
  python -m pipelines.run_cbm_neuro_symbolic_pipeline --config params.yaml
"""
import argparse
import sys
import time

from pipelines import (
    stage1b_train_cbm,
    stage2_extract_features,
    stage3_extract_rules,
    stage4_select_rules_gflownet,
    stage5_train_rule_regularized,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main(config_path: str, skip_stage1b: bool = False) -> None:
    total_start = time.time()
    logger.info("=" * 65)
    logger.info("BẮT ĐẦU PIPELINE TÍCH HỢP CBM + GFLOWNET + RULE REGULARIZATION")
    logger.info("=" * 65)

    # Stage 1b: Train CBM
    if not skip_stage1b:
        logger.info("\n>>> [1/5] STAGE 1b: Huấn luyện CBM với Concept Loss...")
        stage1b_train_cbm.main(config_path)
    else:
        logger.info("\n>>> [1/5] BỎ QUA STAGE 1b (sử dụng checkpoint CBM có sẵn).")

    # Stage 2: Extract Concept Features
    logger.info("\n>>> [2/5] STAGE 2: Trích Concept Probabilities từ CBM...")
    stage2_extract_features.main(config_path, source_override="cbm")

    # Stage 3: Extract Human-interpretable Rules
    logger.info("\n>>> [3/5] STAGE 3: Trích luật ngữ nghĩa (IF Concept THEN Class)...")
    stage3_extract_rules.main(config_path)

    # Stage 4: GFlowNet Rule Selection
    logger.info("\n>>> [4/5] STAGE 4: GFlowNet tuyển chọn tập luật Concept...")
    stage4_select_rules_gflownet.main(config_path)

    # Stage 5: Rule-Regularized Training on CBM
    logger.info("\n>>> [5/5] STAGE 5: Fine-tune CBM với Rule Penalty...")
    stage5_train_rule_regularized.main(config_path, target_override="cbm")

    elapsed = (time.time() - total_start) / 60
    logger.info("\n" + "=" * 65)
    logger.info("HOÀN THÀNH TOÀN BỘ PIPELINE TÍCH HỢP TRONG %.2f PHÚT!", elapsed)
    logger.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    parser.add_argument(
        "--skip-stage1b",
        action="store_true",
        help="Bỏ qua huấn luyện CBM nếu đã có outputs/cbm/cbm_best.pth",
    )
    args = parser.parse_args()
    main(args.config, skip_stage1b=args.skip_stage1b)
