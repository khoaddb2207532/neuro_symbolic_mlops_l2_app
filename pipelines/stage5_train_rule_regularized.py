"""Stage 5 — Fine-tune lại CNN với rule penalty (vectorized) từ luật đã chọn.

Ở stage này model full fine-tune với differential LR cố định và khởi động
lại từ baseline đã hội tụ
(baseline_best.pth từ stage 1), KHÔNG khởi tạo lại từ ImageNet — vì mục tiêu
bây giờ là tinh chỉnh sâu hơn có ràng buộc luật dựa trên những gì baseline đã
học được.
"""
import argparse
import hashlib
import json
import os
import pickle

import torch

from src.data.dataset import create_dataloaders, NeuroSymbolicDataset
from src.evaluation.evaluate import evaluate_model_performance, plot_training_history
from src.models.cbm import ConceptBottleneckModel
from src.models.cnn import ImageClassificationBaseline
from src.rules.rule_types import RuleSet
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


def main(params_path: str, target_override: str = None) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    filtered_dir = os.path.join(params["output_dir"], "04_filtered_rules")
    with open(os.path.join(filtered_dir, "selected_rules.pkl"), "rb") as f:
        selected_rules = pickle.load(f)
    rule_set = RuleSet(rules=selected_rules)

    dataloaders, train_loader, val_loader, test_loader = create_dataloaders(
        params["data_dir"], batch_size=params["batch_size"], num_workers=params["num_workers"], seed=params["seed"]
    )
    class_names = [
        n for n, _ in sorted(NeuroSymbolicDataset(params["data_dir"], "test").class_to_idx.items(), key=lambda x: x[1])
    ]

    save_dir = os.path.join(params["output_dir"], "05_rules_model")
    os.makedirs(save_dir, exist_ok=True)

    target_model = target_override or params.get("rule_penalty", {}).get("target_model", "baseline")
    architecture = selected_baseline_architecture(params)

    if target_model == "cbm":
        cbm_dir = os.path.join(params["output_dir"], "cbm")
        metadata_path = os.path.join(cbm_dir, "cbm_metadata.json")
        concept_names = None
        n_concepts = params.get("cbm", {}).get("n_concepts", 30)
        arch = architecture
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            concept_names = meta.get("concept_names")
            n_concepts = meta.get("n_concepts", n_concepts)
            arch = meta.get("architecture", arch)

        cbm_ckpt = os.path.join(cbm_dir, "cbm_best.pth")
        if not os.path.exists(cbm_ckpt):
            cbm_ckpt = os.path.join(cbm_dir, "final_model_weights.pth")

        model = ConceptBottleneckModel(
            architecture=arch,
            n_concepts=n_concepts,
            num_classes=params["num_classes"],
            pretrained=False,
            concept_names=concept_names,
        )
        load_model_weights(model, cbm_ckpt, device, required=True)
        logger.info("Fine-tune CBM với Rule Penalty (n_concepts=%d) từ %s", n_concepts, cbm_ckpt)
    else:
        model = ImageClassificationBaseline(
            architecture=architecture,
            num_classes=params["num_classes"],
            pretrained=False,
        )
        baseline_ckpt = selected_baseline_checkpoint(params)
        load_model_weights(model, baseline_ckpt, device, required=True)
        logger.info("Fine-tune Baseline CNN với Rule Penalty từ %s", baseline_ckpt)

    train_cfg = {
        "lr_backbone": params["transfer_learning"]["lr_backbone"],
        "lr_head": params["transfer_learning"]["lr_head"],
        "weight_decay": params["weight_decay"],
        "monitor_metric": params.get("monitor_metric", "val_acc"),
        "save_dir": save_dir,
        "resume": bool(params["rule_penalty"].get("resume", False)),
        "resume_checkpoint_path": os.path.join(save_dir, "training_last.pth"),
        "resume_identity": {
            "stage": "stage5_train_rule_regularized",
            "seed": params["seed"],
            "architecture": architecture,
            "gflownet_loss_type": params.get("gflownet", {}).get("loss_type"),
            "selected_rules_sha256": _sha256_file(
                os.path.join(filtered_dir, "selected_rules.pkl")
            ),
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
        rule_set=rule_set,
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

    evaluate_model_performance(model, test_loader, device, class_names, title="Rule-Regularized CNN Performance", output_dir=save_dir)
    plot_training_history(history, save_dir=save_dir, title_suffix="Rule-Regularized CNN")
    logger.info("Stage 5 hoàn thành với baseline '%s'.", architecture)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument(
        "--target",
        choices=["baseline", "cbm"],
        default=None,
        help="Model cần fine-tune với luật: 'baseline' hoặc 'cbm'",
    )
    args = parser.parse_args()
    main(args.config, target_override=args.target)
