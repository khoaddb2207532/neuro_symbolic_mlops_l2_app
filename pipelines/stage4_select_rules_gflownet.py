"""DVC Stage 4 — Dùng GFlowNet để chọn tập luật con tối ưu."""
import argparse
import os
import pickle

import torch

from src.gflownet.pipeline import RuleExtractionPipeline
from src.gflownet.uncertainty import compute_prediction_stats_from_logits, compute_sample_weight
from src.rules.validator import RuleValidator
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

    with open(os.path.join(rules_dir, "raw_rules.pkl"), "rb") as f:
        raw_rules = pickle.load(f)

    train_features = torch.load(f"{features_dir}/train_features.pt").to(device)
    train_labels = torch.load(f"{features_dir}/train_labels.pt").to(device)
    val_features = torch.load(f"{features_dir}/val_features.pt").to(device)
    val_labels = torch.load(f"{features_dir}/val_labels.pt").to(device)

    # Lọc luật + build cover/correct/rule_len TRONG MỘT LẦN QUÉT DUY NHẤT.
    # Các tensor này được dùng lại y hệt bởi RuleSetReward — không quét lại
    # val set lần hai trong pipeline.
    validator = RuleValidator(
        min_supp=params["rules"]["min_support"],
    )
    valid_rule_set, cover, correct, rule_len = validator.validate_and_build_tensors(
        raw_rules,
        train_features,
        train_labels,
        store_device=device,
        num_classes=params["num_classes"],
    )
    with open(os.path.join(output_dir, "valid_rules.pkl"), "wb") as f:
        pickle.dump(valid_rule_set, f)

    valid_rules = list(valid_rule_set.rules)
    logger.info("Số luật hợp lệ sau khi lọc trên training_data: %d", len(valid_rules))

    # ------------------------------------------------------------------
    # sample_weight (u) — độ không chắc chắn/lỗi của CNN trên ĐÚNG
    # val_features/val_labels đã dùng để build cover/correct ở trên.
    #
    # KHÔNG forward lại CNN: nạp thẳng `train_logits.pt` đã được stage2 lưu
    # sẵn cùng lúc với val_features.pt (xem features.py::extract_and_save_
    # features) — cùng một lần forward, cùng checkpoint, cùng thứ tự mẫu,
    # loại bỏ hoàn toàn rủi ro lệch checkpoint hoặc lệch kiến trúc so với
    # lúc trích features cho RF.
    #
    # Biến thể D:  u = u_error + lam * (1 - u_error) * u_entropy_normalized
    # Xem src/gflownet/uncertainty.py và src/gflownet/reward.py để biết
    # cách u chỉ tác động tới `f_cover` (coverage weighted), không đụng
    # `f_quality`/`f_overlap`.
    # ------------------------------------------------------------------
    uc_cfg = params.get("uncertainty", {})
    sample_weight = None
    if uc_cfg.get("enabled", False):
        num_classes = params["num_classes"]
        train_logits = torch.load(f"{features_dir}/train_logits.pt").to(device)
        assert train_logits.shape[0] == train_labels.shape[0], (
            f"train_logits ({train_logits.shape[0]} hàng) không khớp train_labels "
            f"({train_labels.shape[0]} hàng) — kiểm tra lại stage2 đã lưu logits "
            "đúng cùng lần forward với features/labels chưa."
        )

        pred, entropy_norm = compute_prediction_stats_from_logits(
            logits=train_logits, num_classes=num_classes,
        )
        sample_weight = compute_sample_weight(
            pred=pred,
            entropy_norm=entropy_norm,
            y_true=train_labels.cpu(),
            lam=uc_cfg.get("lam", 0.3),
            clip_max=uc_cfg.get("clip_max", None),
        )
        logger.info(
            "sample_weight (u): n=%d, mean=%.4f, frac(u>0.5)=%.4f",
            sample_weight.shape[0],
            sample_weight.mean().item(),
            (sample_weight > 0.5).float().mean().item(),
        )
        with open(os.path.join(output_dir, "sample_weight.pkl"), "wb") as f:
            pickle.dump(sample_weight, f)
    else:
        logger.info("uncertainty.enabled = False — dùng coverage thô (không weighted).")

    gfn_cfg = params["gflownet"]
    pipeline = RuleExtractionPipeline(
        device=device,
        alpha=gfn_cfg.get("alpha", 1.0),
        lambda_1=gfn_cfg.get("lambda_1", 1.0),
        lambda_2=gfn_cfg.get("lambda_2", 1.0),
        lambda_3=gfn_cfg.get("lambda_3", 1.0),
        K=gfn_cfg.get("K", gfn_cfg.get("max_rules", 10)),
        gamma=gfn_cfg.get("gamma", 3.0),
        grad_clip_max_norm=gfn_cfg.get("grad_clip_max_norm", 5.0),
    )
    selected_rules = pipeline.run(
        valid_rules=valid_rules,
        cover=cover,
        correct=correct,
        rule_len=rule_len,
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
        sample_weight=sample_weight,
        replay_capacity=gfn_cfg.get("replay_capacity", 10000),
        num_candidates=gfn_cfg.get("num_candidates", 100),
        # cover_test/correct_test: chưa nối vào đây vì stage4 hiện chỉ nạp
        # train_features/val_features (không có tensor test cùng trục LUẬT
        # với valid_rules) — nếu muốn giai đoạn 4 (post-training) của pipeline
        # chọn tập luật theo holdout TEST thay vì tạm dùng val, cần build
        # thêm cover_test/correct_test (cùng valid_rule_set, KHÔNG re-filter)
        # từ test_features/test_labels rồi truyền vào đây.
    )

    with open(os.path.join(output_dir, "selected_rules.pkl"), "wb") as f:
        pickle.dump(selected_rules, f)
    save_rules_excel(selected_rules, os.path.join(output_dir, "selected_rules.xlsx"))
    logger.info("Stage 4 hoàn thành. %d luật được GFlowNet chọn.", len(selected_rules))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)