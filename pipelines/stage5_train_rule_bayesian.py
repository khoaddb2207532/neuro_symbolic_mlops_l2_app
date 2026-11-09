"""DVC Stage 5 (biến thể) — Fine-tune CNN với rule-penalty kiểu BAYESIAN
MARGINALIZATION thay vì phạt theo 1 tập luật cố định.

Khác biệt so với `stage5_train_rule_regularized.py`:
  1. KHÔNG dùng `selected_rules.pkl` (tập luật MAP/best-found cố định của
     GFlowNet) làm rule_set để phạt.
  2. Thay vào đó, nạp policy sampler GFlowNet đã train cùng
     `gflownet_rule_order.pkl`, cả hai đều do
     `RuleExtractionPipeline.run()` ở stage4 lưu ra — KHÔNG train lại
     GFlowNet ở đây), dùng làm FROZEN SAMPLER.
  3. Mỗi bước train CNN, resample K tập luật MỚI từ sampler này và tính
     rule-penalty kỳ vọng (Monte Carlo không chệch qua K mẫu) — xem
     `src.rules.bayesian_penalty.BayesianRuleMarginalization`.
Vẫn dùng chung `train_model()`/`train_one_epoch()` ở `src/training/trainer.py`
— chỉ khác ở CÁCH XÂY DỰNG penalty_module (truyền thẳng vào thay vì để
`train_model` tự build từ `rule_set`).
"""
import argparse
import hashlib
import os

import torch

from src.data.dataset import create_dataloaders, NeuroSymbolicDataset
from src.evaluation.evaluate import evaluate_model_performance, plot_training_history
from src.gflownet.rule_ranking_analysis import load_rule_order, rebuild_gflownet
from src.models.cnn import ImageClassificationBaseline
from src.rules.bayesian_penalty import BayesianRuleMarginalization
from src.training.trainer import train_model
from src.utils.checkpoint import load_model_weights
from src.utils.config import (
    load_params,
    selected_baseline_architecture,
    selected_baseline_checkpoint,
)
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(params_path: str, variant: str = "diverse") -> None:
    if variant not in {"diverse", "elite"}:
        raise ValueError(f"Bayesian variant không hợp lệ: {variant}")
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    filtered_dir = os.path.join(params["output_dir"], "04_filtered_rules")
    bayes_cfg = dict(params.get("rule_penalty_bayesian", {}))
    if variant == "elite":
        bayes_cfg.update(params.get("rule_penalty_bayesian_elite", {}))
    default_checkpoint = (
        "gflownet_best_elite.pth"
        if variant == "elite"
        else "gflownet_best_diverse.pth"
    )
    checkpoint_name = bayes_cfg.get("sampler_checkpoint", default_checkpoint)
    ckpt_path = os.path.join(filtered_dir, checkpoint_name)
    if not os.path.exists(ckpt_path) and variant != "elite":
        fallback_name = "gflownet_best_converged.pth"
        logger.warning(
            "Chưa có %s — fallback sang %s.",
            checkpoint_name,
            fallback_name,
        )
        ckpt_path = os.path.join(filtered_dir, fallback_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Không tìm thấy elite sampler checkpoint: {ckpt_path}. "
            "Hãy chạy lại Stage 4 để tạo gflownet_best_elite.pth."
        )

    elite_obj = None
    if variant == "elite":
        elite_obj = torch.load(ckpt_path, map_location="cpu")
        if elite_obj.get("checkpoint_role") != "elite":
            raise ValueError(
                f"{ckpt_path} không phải elite checkpoint "
                f"(role={elite_obj.get('checkpoint_role')!r})."
            )

    # ---- 1) Nạp lại policy GFlowNet đã train làm FROZEN SAMPLER (không
    # train lại bất kỳ tham số nào của nó) ----
    rule_order = load_rule_order(filtered_dir)
    gflownet, env, valid_rules, _reward_module = rebuild_gflownet(rule_order, ckpt_path, device)
    logger.info(
        "Đã nạp frozen GFlowNet sampler từ %s (%d luật, loss_type=%s).",
        ckpt_path, len(valid_rules), rule_order["loss_type"],
    )

    K = bayes_cfg.get("K", 32)

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
        temp_warmup_epochs=params["rule_penalty"].get("temp_warmup_epochs", 2),
        temp_anneal_epochs=params["rule_penalty"].get("temp_anneal_epochs", 10),
    )

    dataloaders, train_loader, val_loader, test_loader = create_dataloaders(
        params["data_dir"], batch_size=params["batch_size"], num_workers=params["num_workers"], seed=params["seed"]
    )
    class_names = [
        n for n, _ in sorted(NeuroSymbolicDataset(params["data_dir"], "test").class_to_idx.items(), key=lambda x: x[1])
    ]

    save_subdir = (
        "05b_rules_model_bayesian_elite"
        if variant == "elite"
        else "05b_rules_model_bayesian"
    )
    save_dir = os.path.join(params["output_dir"], save_subdir)
    os.makedirs(save_dir, exist_ok=True)

    architecture = selected_baseline_architecture(params)
    model = ImageClassificationBaseline(
        architecture=architecture,
        num_classes=params["num_classes"],
        pretrained=False,
    )

    baseline_ckpt = selected_baseline_checkpoint(params)
    load_model_weights(model, baseline_ckpt, device, required=True)

    train_cfg = {
        "lr_backbone": params["transfer_learning"]["lr_backbone"],
        "lr_head": params["transfer_learning"]["lr_head"],
        "weight_decay": params["weight_decay"],
        "monitor_metric": params.get("monitor_metric", "val_acc"),
        "dvclive_path": os.path.join(
            save_dir,
            (
                "dvclive_rule_regularized_bayesian_elite"
                if variant == "elite"
                else "dvclive_rule_regularized_bayesian"
            ),
        ),
        "save_dir": save_dir,
        "resume": bool(bayes_cfg.get("resume", variant == "elite")),
        "resume_checkpoint_path": os.path.join(save_dir, "training_last.pth"),
        "resume_identity": {
            "stage": "stage5_train_rule_bayesian",
            "variant": variant,
            "seed": params["seed"],
            "architecture": architecture,
            "sampler_checkpoint": checkpoint_name,
            "sampler_sha256": _sha256_file(ckpt_path),
            "rule_order_sha256": _sha256_file(
                os.path.join(filtered_dir, "gflownet_rule_order.pkl")
            ),
            "sampler_iteration": (
                int(elite_obj["iteration"]) if elite_obj is not None else None
            ),
            "K": K,
            "batch_size": params["batch_size"],
            "optimizer": {
                "lr_backbone": params["transfer_learning"]["lr_backbone"],
                "lr_head": params["transfer_learning"]["lr_head"],
                "weight_decay": params["weight_decay"],
            },
            "rule_penalty": dict(params["rule_penalty"]),
        },
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
        temp_warmup_epochs=params["rule_penalty"].get("temp_warmup_epochs", 2),
        temp_anneal_epochs=params["rule_penalty"].get("temp_anneal_epochs", 10),
        min_epochs_before_early_stop=params["rule_penalty"].get("min_epochs_before_early_stop", 12),
        num_classes=params["num_classes"],
        config=train_cfg,
    )

    sampling_summary = penalty_module.save_sampling_summary(save_dir)
    logger.info(
        "Đã lưu %d/%d luật từng được GFlowNet sample từ %d ruleset vào %s.",
        sampling_summary["n_rules_ever_sampled"],
        sampling_summary["n_valid_rules"],
        sampling_summary["total_sampled_rulesets"],
        save_dir,
    )

    for item in sampling_summary["top_impact_rules"][:5]:
        logger.info(
            "Rule impact #%d: index=%d | contribution=%.6f | share=%.2f%%",
            item["rank"],
            item["rule_index"],
            item["cumulative_penalty_contribution"],
            item["contribution_share"] * 100.0,
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
    parser.add_argument(
        "--variant", choices=["diverse", "elite"], default="diverse"
    )
    args = parser.parse_args()
    main(args.config, variant=args.variant)
