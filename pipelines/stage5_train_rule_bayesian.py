"""DVC Stage 5 (biến thể) — Fine-tune CNN với rule-penalty kiểu BAYESIAN
MARGINALIZATION thay vì phạt theo 1 tập luật cố định.

Khác biệt so với `stage5_train_rule_regularized.py`:
  1. KHÔNG dùng `selected_rules.pkl` (tập luật MAP/best-found cố định của
     GFlowNet) làm rule_set để phạt.
  2. Thay vào đó, nạp lại CHÍNH policy GFlowNet đã train (`gflownet_best.pth`
     + `gflownet_rule_order.pkl`, cả hai đều do
     `RuleExtractionPipeline.run()` ở stage4 lưu ra — KHÔNG train lại
     GFlowNet ở đây), dùng làm FROZEN SAMPLER.
  3. Mỗi bước train CNN, resample K tập luật MỚI từ sampler này và tính
     rule-penalty kỳ vọng (Monte Carlo không chệch qua K mẫu) — xem
     `src.rules.bayesian_penalty.BayesianRuleMarginalization`.
  4. Bật `resolve_loss_conflict=True`: xử lý xung đột gradient (PCGrad) giữa
     CE loss và rule-penalty loss (xem `src.training.trainer._resolve_loss_conflict`).

Vẫn dùng chung `train_model()`/`train_one_epoch()` ở `src/training/trainer.py`
— chỉ khác ở CÁCH XÂY DỰNG penalty_module (truyền thẳng vào thay vì để
`train_model` tự build từ `rule_set`) và cờ cấu hình `resolve_loss_conflict`.
"""
import argparse
import os

import torch

from src.data.dataset import create_dataloaders, NeuroSymbolicDataset
from src.evaluation.evaluate import evaluate_model_performance, plot_training_history
from src.gflownet.rule_ranking_analysis import load_rule_order, rebuild_gflownet
from src.models.cnn import build_selected_baseline, selected_baseline_checkpoint
from src.rules.bayesian_penalty import BayesianRuleMarginalization
from src.training.trainer import train_model
from src.utils.checkpoint import load_model_weights
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main(params_path: str) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    filtered_dir = os.path.join(params["output_dir"], "04_filtered_rules")

    # ---- 1) Nạp lại policy GFlowNet đã train làm FROZEN SAMPLER (không
    # train lại bất kỳ tham số nào của nó) ----
    # LƯU Ý: pipeline huấn luyện mới (src/gflownet/pipeline.py) đã bỏ
    # `_SamplerCheckpointTracker`/checkpoint riêng `gflownet_best_sampler.pth`
    # (vốn chỉ lưu khi diversity không dưới ngưỡng tối thiểu). Giờ chỉ còn
    # DUY NHẤT `gflownet_best.pth` — lưu tại điểm val_loss thấp nhất, với
    # mode-collapse đã được early-stopping chặn ngay trong lúc train (xem
    # `_check_early_stopping`) — nên dùng thẳng checkpoint này làm sampler,
    # không cần fallback nữa.
    ckpt_path = os.path.join(filtered_dir, "gflownet_best.pth")

    rule_order = load_rule_order(filtered_dir)
    gflownet, env, valid_rules, _reward_module = rebuild_gflownet(rule_order, ckpt_path, device)
    logger.info(
        "Đã nạp frozen GFlowNet sampler từ %s (%d luật, loss_type=%s).",
        ckpt_path, len(valid_rules), rule_order["loss_type"],
    )

    bayes_cfg = params.get("rule_penalty_bayesian", {})
    K = bayes_cfg.get("K", 32)

    features_dir = os.path.join(params["output_dir"], "02_features")
    val_features = torch.load(os.path.join(features_dir, "val_features.pt"), map_location="cpu")
    val_labels = torch.load(os.path.join(features_dir, "val_labels.pt"), map_location="cpu")

    penalty_module = BayesianRuleMarginalization(
        valid_rules=valid_rules,
        gflownet=gflownet,
        env=env,
        K=K,
        penalty_weight=params["rule_penalty"]["weight"],
        use_confidence=True,
        smoothing=params["rule_penalty"]["smoothing"],
        num_classes=params["num_classes"],
        initial_temp=params["rule_penalty"]["initial_temp"],
        final_temp=params["rule_penalty"]["final_temp"],
        validation_features=val_features,
        validation_labels=val_labels,
        min_rule_confidence=bayes_cfg.get("min_rule_confidence", 0.7),
        min_val_support=bayes_cfg.get("min_val_support", 0.01),
        min_val_precision=bayes_cfg.get("min_val_precision", 0.7),
        cnn_uncertainty_threshold=bayes_cfg.get("cnn_uncertainty_threshold", 0.75),
    )
    logger.info(
        "Rule validation gate: giữ %d/%d luật (confidence>=%.2f, val_support>=%.3f, val_precision>=%.2f).",
        penalty_module.num_eligible_rules, len(valid_rules),
        bayes_cfg.get("min_rule_confidence", 0.7),
        bayes_cfg.get("min_val_support", 0.01),
        bayes_cfg.get("min_val_precision", 0.7),
    )

    dataloaders, train_loader, val_loader, test_loader = create_dataloaders(
        params["data_dir"], batch_size=params["batch_size"], num_workers=params["num_workers"], seed=params["seed"]
    )
    class_names = [
        n for n, _ in sorted(NeuroSymbolicDataset(params["data_dir"], "test").class_to_idx.items(), key=lambda x: x[1])
    ]

    save_dir = os.path.join(params["output_dir"], "05b_rules_model_bayesian")
    os.makedirs(save_dir, exist_ok=True)

    model = build_selected_baseline(params, pretrained=False)

    # Giống stage5 gốc: tiếp tục từ baseline đã hội tụ (stage1), không phải
    # từ ImageNet.
    baseline_ckpt = selected_baseline_checkpoint(params)
    load_model_weights(model, baseline_ckpt, device, required=True)

    fine_tune_lr_backbone = bayes_cfg.get("fine_tune_lr_backbone", 1e-6)
    train_cfg = {
        # Trainer hiện dùng một LR cho toàn bộ CNN; Bayesian giữ mức fine-tune
        # nhỏ đã cấu hình thay vì vô tình rơi về mặc định 1e-3.
        "lr": fine_tune_lr_backbone,
        "weight_decay": params["weight_decay"],
        "monitor_metric": params.get("monitor_metric", "val_acc"),
        "use_scheduler": True,
        "scheduler_factor": 0.1,
        "scheduler_patience": 3,
        "dvclive_path": os.path.join(save_dir, "dvclive_rule_regularized_bayesian"),
        "save_dir": save_dir,
        # ---- Task 3: xử lý xung đột gradient CE vs rule-penalty (PCGrad) ----
        "resolve_loss_conflict": bayes_cfg.get("resolve_loss_conflict", True),
        "penalty_warmup_start_epoch":
            params["rule_penalty"]["warmup_start_epoch"],
        "penalty_warmup_end_epoch":
            params["rule_penalty"]["warmup_end_epoch"],
        "temperature_end_epoch":
            params["rule_penalty"]["temperature_end_epoch"],
        "intermediate_temp":
            params["rule_penalty"].get("intermediate_temp"),
    }

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        rule_set=None,                 # không dùng — penalty_module đã được truyền thẳng
        penalty_module=penalty_module,  # Bayesian marginalization qua frozen sampler
        num_epochs=params["num_epochs"],
        patience=params["patience"],
        device=device,
        penalty_weight=params["rule_penalty"]["weight"],
        use_confidence=True,
        smoothing=params["rule_penalty"]["smoothing"],
        initial_temp=params["rule_penalty"]["initial_temp"],
        final_temp=params["rule_penalty"]["final_temp"],
        num_classes=params["num_classes"],
        config=train_cfg,
    )

    evaluate_model_performance(
        model, test_loader, device, class_names,
        title="Rule-Regularized CNN Performance (Bayesian marginalization)", output_dir=save_dir,
    )
    plot_training_history(history, save_dir=save_dir, title_suffix="Rule-Regularized CNN (Bayesian)")
    logger.info("Stage 5 (Bayesian marginalization) hoàn thành.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
