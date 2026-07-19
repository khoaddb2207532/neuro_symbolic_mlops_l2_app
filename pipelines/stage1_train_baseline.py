"""DVC Stage 1 — Huấn luyện CNN baseline với chiến lược progressive unfreezing.

Chạy: python -m pipelines.stage1_train_baseline --config params.yaml
"""
import argparse
import json
import os

import torch

from src.data.dataset import create_dataloaders, NeuroSymbolicDataset
from src.data.protocol import validate_audit_summary, validate_split_protocol
from src.evaluation.evaluate import evaluate_classification_metrics, evaluate_model_performance, plot_training_history
from src.models.cnn import CNNBaseline
from src.training.trainer import train_model
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main(params_path: str) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    audit_path = os.path.join(params["output_dir"], "00_data_audit", "summary.json")
    if not os.path.exists(audit_path):
        raise FileNotFoundError(f"required data audit is missing: {audit_path}")
    with open(audit_path, encoding="utf-8") as stream:
        audit_summary = json.load(stream)
    audit_warning = None
    try:
        validate_audit_summary(
            audit_summary,
            fail_on_near_duplicate_leakage=params["data_validation"].get(
                "fail_on_near_duplicate_leakage", True
            ),
        )
    except ValueError as error:
        # Stage 1 is deliberately warning-only for audit findings.  Keep the
        # warning in both the console log and immutable baseline report so the
        # resulting metrics cannot be mistaken for leakage-free measurements.
        audit_warning = str(error)
        logger.warning("DATA AUDIT WARNING (training will continue): %s", audit_warning)

    dataloaders, train_loader, val_loader, test_loader = create_dataloaders(
        params["data_dir"], batch_size=params["batch_size"], num_workers=params["num_workers"], seed=params["seed"]
    )
    class_names = [
        n for n, _ in sorted(NeuroSymbolicDataset(params["data_dir"], "test").class_to_idx.items(), key=lambda x: x[1])
    ]
    protocol = params["data_validation"]
    split_report = validate_split_protocol(
        {name: len(loader.dataset) for name, loader in dataloaders.items()},
        [label for _, label in val_loader.dataset.samples],
        min_val_fraction=protocol["min_val_fraction"],
        min_val_samples_per_class=protocol["min_val_samples_per_class"],
        expected_classes=range(params["num_classes"]),
    )

    save_dir = os.path.join(params["output_dir"], "01_baseline")
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join("reports", "baseline_metrics.json")
    if os.path.exists(metrics_path):
        raise FileExistsError(
            f"{metrics_path} already exists and is immutable; archive it explicitly before retraining baseline"
        )

    model = CNNBaseline(num_classes=params["num_classes"], freeze_stage="head_only")

    # freeze_schedule đến trực tiếp từ params.yaml -> đây là "chiến lược transfer
    # learning" được cấu hình khai báo (declarative), không hard-code trong code.
    freeze_schedule = {int(k): v for k, v in params["transfer_learning"]["freeze_schedule"].items()}

    train_cfg = {
        "lr_backbone": params["transfer_learning"]["lr_backbone"],
        "lr_head": params["transfer_learning"]["lr_head"],
        "weight_decay": params["weight_decay"],
        "freeze_bn": params["transfer_learning"]["freeze_bn"],
        "freeze_schedule": freeze_schedule,
        "monitor_metric": params.get("monitor_metric", "val_acc"),
        "use_scheduler": True,
        "scheduler_factor": 0.1,
        "scheduler_patience": 3,
        "dvclive_path": os.path.join(save_dir, "dvclive_baseline"),
        "save_dir": save_dir,
    }

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        rule_set=None,
        num_epochs=params["num_epochs"],
        patience=params["patience"],
        device=device,
        penalty_weight=0.0,
        num_classes=params["num_classes"],
        config=train_cfg,
    )

    val_metrics = evaluate_classification_metrics(model, val_loader, device)
    test_metrics = evaluate_classification_metrics(model, test_loader, device)
    evaluate_model_performance(model, val_loader, device, class_names, title="Baseline CNN Validation", output_dir=save_dir)
    evaluate_model_performance(model, test_loader, device, class_names, title="Baseline CNN Test", output_dir=save_dir)
    plot_training_history(history, save_dir=save_dir, title_suffix="Baseline CNN")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "cnn_baseline.pt"))
    with open(metrics_path, "x", encoding="utf-8") as stream:
        json.dump({"validation": val_metrics, "test": test_metrics,
                   "data_protocol": split_report, "data_audit_warning": audit_warning,
                   "seed": params["seed"]}, stream, indent=2)
    logger.info("Stage 1 hoàn thành. Checkpoint tại %s", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
