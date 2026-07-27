"""DVC Stage 1 — Huấn luyện CNN baseline bằng full fine-tuning.

Chạy: python -m pipelines.stage1_train_baseline --config params.yaml
"""
import argparse
import os

import torch

from src.data.dataset import create_dataloaders, NeuroSymbolicDataset
from src.evaluation.evaluate import evaluate_model_performance, plot_training_history
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

    dataloaders, train_loader, val_loader, test_loader = create_dataloaders(
        params["data_dir"], batch_size=params["batch_size"], num_workers=params["num_workers"], seed=params["seed"]
    )
    class_names = [
        n for n, _ in sorted(NeuroSymbolicDataset(params["data_dir"], "test").class_to_idx.items(), key=lambda x: x[1])
    ]

    save_dir = os.path.join(params["output_dir"], "01_baseline")
    os.makedirs(save_dir, exist_ok=True)

    model = CNNBaseline(num_classes=params["num_classes"])

    train_cfg = {
        "lr_backbone": params["transfer_learning"]["lr_backbone"],
        "lr_head": params["transfer_learning"]["lr_head"],
        "weight_decay": params["weight_decay"],
        "monitor_metric": params.get("monitor_metric", "val_acc"),
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

    evaluate_model_performance(model, test_loader, device, class_names, title="Baseline CNN Performance", output_dir=save_dir)
    plot_training_history(history, save_dir=save_dir, title_suffix="Baseline CNN")
    logger.info("Stage 1 hoàn thành. Checkpoint tại %s", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
