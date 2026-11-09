"""DVC Stage 5 — distill selected rules sang CNN student pretrained ImageNet.

Teacher nạp baseline checkpoint và luôn bị đóng băng. Student dùng
``pretrained=True`` và tuyệt đối không nạp baseline checkpoint.
"""
import argparse
import os
import pickle

import torch

from src.data.dataset import create_dataloaders, NeuroSymbolicDataset
from src.evaluation.evaluate import evaluate_model_performance, plot_training_history
from src.models.cnn import ImageClassificationBaseline
from src.rules.distillation import RuleDistillationPenalty
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


def main(params_path: str) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    filtered_dir = os.path.join(params["output_dir"], "04_filtered_rules")
    with open(os.path.join(filtered_dir, "selected_rules.pkl"), "rb") as file:
        rule_set = RuleSet(rules=pickle.load(file))

    _, train_loader, val_loader, test_loader = create_dataloaders(
        params["data_dir"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        seed=params["seed"],
    )
    class_names = [
        name
        for name, _ in sorted(
            NeuroSymbolicDataset(params["data_dir"], "test").class_to_idx.items(),
            key=lambda item: item[1],
        )
    ]

    architecture = selected_baseline_architecture(params)
    teacher = ImageClassificationBaseline(
        architecture=architecture,
        num_classes=params["num_classes"],
        pretrained=False,
    )
    load_model_weights(
        teacher, selected_baseline_checkpoint(params), device, required=True
    )

    # Student chỉ kế thừa ImageNet, không kế thừa checkpoint baseline.
    student = ImageClassificationBaseline(
        architecture=architecture,
        num_classes=params["num_classes"],
        pretrained=True,
    )

    distill_cfg = params["rule_distillation"]
    rule_cfg = params["rule_penalty"]
    penalty_module = RuleDistillationPenalty(
        teacher=teacher,
        rule_set=rule_set,
        num_classes=params["num_classes"],
        penalty_weight=distill_cfg["weight"],
        distillation_temperature=distill_cfg["temperature"],
        initial_temp=rule_cfg["initial_temp"],
        final_temp=rule_cfg["final_temp"],
        temp_warmup_epochs=rule_cfg.get("temp_warmup_epochs", 2),
        temp_anneal_epochs=rule_cfg.get("temp_anneal_epochs", 10),
        use_confidence=distill_cfg.get("use_confidence", True),
    )

    save_dir = os.path.join(params["output_dir"], "05c_rules_distillation")
    train_cfg = {
        "lr_backbone": params["transfer_learning"]["lr_backbone"],
        "lr_head": params["transfer_learning"]["lr_head"],
        "weight_decay": params["weight_decay"],
        "monitor_metric": params.get("monitor_metric", "val_acc"),
        "dvclive_path": os.path.join(save_dir, "dvclive_rule_distillation"),
        "save_dir": save_dir,
    }
    student, history = train_model(
        model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        rule_set=None,
        penalty_module=penalty_module,
        num_epochs=params["num_epochs"],
        patience=params["patience"],
        device=device,
        penalty_weight=distill_cfg["weight"],
        smoothing=rule_cfg["smoothing"],
        initial_temp=rule_cfg["initial_temp"],
        final_temp=rule_cfg["final_temp"],
        temp_warmup_epochs=rule_cfg.get("temp_warmup_epochs", 2),
        temp_anneal_epochs=rule_cfg.get("temp_anneal_epochs", 10),
        min_epochs_before_early_stop=rule_cfg.get(
            "min_epochs_before_early_stop", 12
        ),
        num_classes=params["num_classes"],
        config=train_cfg,
    )

    evaluate_model_performance(
        student,
        test_loader,
        device,
        class_names,
        title="Rule Distillation CNN Performance",
        output_dir=save_dir,
    )
    plot_training_history(
        history, save_dir=save_dir, title_suffix="Rule Distillation CNN"
    )
    logger.info(
        "Stage 5 distillation hoàn thành: student '%s' khởi tạo từ ImageNet.",
        architecture,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    arguments = parser.parse_args()
    main(arguments.config)
