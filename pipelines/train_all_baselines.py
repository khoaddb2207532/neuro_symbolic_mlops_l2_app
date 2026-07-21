"""Train multiple torchvision baselines sequentially in one process.

Usage:
    python -m pipelines.train_all_baselines --config params.yaml
    python -m pipelines.train_all_baselines --models resnet50 vit
"""
import argparse
import gc
import json
import os
from datetime import datetime, timezone
from typing import Iterable

import torch

from src.data.dataset import NeuroSymbolicDataset, create_dataloaders
from src.data.protocol import validate_audit_summary, validate_split_protocol
from src.evaluation.evaluate import (
    evaluate_classification_metrics,
    evaluate_model_performance,
    plot_training_history,
)
from src.models.cnn import VisionBaseline, canonical_baseline_name
from src.training.trainer import train_model
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def train_all(params_path: str, requested_models: Iterable[str] | None = None) -> dict:
    params = load_params(params_path)
    baseline_cfg = params.get("baselines", {})
    raw_models = list(requested_models or baseline_cfg.get("models", []))
    if not raw_models:
        raise ValueError("Configure baselines.models or pass --models.")
    model_names = [canonical_baseline_name(name) for name in raw_models]
    if len(model_names) != len(set(model_names)):
        raise ValueError("The baseline list contains duplicate models (possibly via aliases).")

    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audit_path = os.path.join(params["output_dir"], "00_data_audit", "summary.json")
    if not os.path.exists(audit_path):
        raise FileNotFoundError(f"Required data audit is missing: {audit_path}")
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
        audit_warning = str(error)
        logger.warning("DATA AUDIT WARNING (training will continue): %s", audit_warning)

    dataloaders, train_loader, val_loader, test_loader = create_dataloaders(
        params["data_dir"], params["batch_size"], params["num_workers"], params["seed"]
    )
    test_dataset = NeuroSymbolicDataset(params["data_dir"], "test")
    class_names = [name for name, _ in sorted(test_dataset.class_to_idx.items(), key=lambda x: x[1])]
    protocol = params["data_validation"]
    split_report = validate_split_protocol(
        {name: len(loader.dataset) for name, loader in dataloaders.items()},
        [label for _, label in val_loader.dataset.samples],
        min_val_fraction=protocol["min_val_fraction"],
        min_val_samples_per_class=protocol["min_val_samples_per_class"],
        expected_classes=range(params["num_classes"]),
    )

    root_dir = os.path.join(params["output_dir"], "01_baselines")
    summary_path = os.path.join(root_dir, "summary.json")
    summary = {
        "status": "running", "device": device, "models": model_names,
        "started_at": datetime.now(timezone.utc).isoformat(), "results": {},
    }
    _write_json(summary_path, summary)

    for index, name in enumerate(model_names, start=1):
        logger.info("Baseline %d/%d: %s", index, len(model_names), name)
        set_seed(params["seed"])
        save_dir = os.path.join(root_dir, name)
        model = VisionBaseline(name, params["num_classes"], pretrained=baseline_cfg.get("pretrained", True))
        train_cfg = {
            "lr": baseline_cfg.get("learning_rate", params["learning_rate"]),
            "weight_decay": params["weight_decay"],
            "monitor_metric": params.get("monitor_metric", "val_acc"),
            "use_scheduler": True,
            "scheduler_factor": 0.1,
            "scheduler_patience": 3,
            "dvclive_path": os.path.join(save_dir, "dvclive"),
            "save_dir": save_dir,
        }
        model, history = train_model(
            model, train_loader, val_loader, rule_set=None,
            num_epochs=baseline_cfg.get("num_epochs", params["num_epochs"]),
            patience=baseline_cfg.get("patience", params["patience"]),
            device=device, penalty_weight=0.0, num_classes=params["num_classes"],
            config=train_cfg,
        )
        val_metrics = evaluate_classification_metrics(model, val_loader, device)
        test_metrics = evaluate_classification_metrics(model, test_loader, device)
        title = name.replace("_", " ").title()
        evaluate_model_performance(model, val_loader, device, class_names, f"{title} Validation", save_dir)
        evaluate_model_performance(model, test_loader, device, class_names, f"{title} Test", save_dir)
        plot_training_history(history, save_dir=save_dir, title_suffix=title)

        checkpoint_path = os.path.join(save_dir, "model.pt")
        torch.save(model.state_dict(), checkpoint_path)
        result = {
            "validation": val_metrics, "test": test_metrics,
            "feature_dim": model.feature_dim, "checkpoint": checkpoint_path,
            "epochs_trained": len(history["train_loss"]), "seed": params["seed"],
            "data_protocol": split_report, "data_audit_warning": audit_warning,
        }
        _write_json(os.path.join(save_dir, "metrics.json"), result)
        summary["results"][name] = result
        _write_json(summary_path, summary)  # preserve completed results if a later model fails
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary["status"] = "completed"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(summary_path, summary)
    logger.info("All baselines completed. Summary: %s", summary_path)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--models", nargs="+", help="Override baselines.models from config")
    args = parser.parse_args()
    train_all(args.config, args.models)
