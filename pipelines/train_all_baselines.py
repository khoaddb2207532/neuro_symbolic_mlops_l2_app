"""Huấn luyện tuần tự nhiều image-classification baseline trong một lần chạy.

Chạy:
    python -m pipelines.train_all_baselines --config params.yaml

Có thể ghi đè danh sách/cấu hình:
    python -m pipelines.train_all_baselines --config params.yaml \
        --models mobilenetv3_small shufflenet_v2_x1_0 resnet50 densenet121 \
        efficientnet_b0 alexnet swin_t vit_b_16 vit_b_32 --epochs 30
"""
import argparse
import gc
import json
import os
import time
from typing import List, Optional

import torch

from src.data.dataset import NeuroSymbolicDataset, create_dataloaders
from src.evaluation.evaluate import evaluate_model_performance, plot_training_history
from src.models.cnn import (
    BASELINE_ARCHITECTURES,
    ImageClassificationBaseline,
    normalize_architecture_name,
)
from src.training.trainer import train_model
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _class_names(data_dir: str) -> List[str]:
    dataset = NeuroSymbolicDataset(data_dir, "test")
    return [
        name
        for name, _ in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])
    ]


def train_all_baselines(
    params_path: str,
    architectures: Optional[List[str]] = None,
    epochs: Optional[int] = None,
) -> List[dict]:
    params = load_params(params_path)
    comparison_cfg = params.get("baseline_comparison", {})
    architectures = architectures or comparison_cfg.get(
        "architectures", list(BASELINE_ARCHITECTURES)
    )
    architectures = [
        normalize_architecture_name(architecture)
        for architecture in architectures
    ]
    invalid_architectures = [
        architecture
        for architecture in architectures
        if architecture not in BASELINE_ARCHITECTURES
    ]
    if invalid_architectures:
        raise ValueError(
            f"Model không được hỗ trợ: {', '.join(invalid_architectures)}. "
            f"Các lựa chọn: {', '.join(BASELINE_ARCHITECTURES)}"
        )
    num_epochs = epochs if epochs is not None else params["num_epochs"]
    pretrained = comparison_cfg.get("pretrained", True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(params["seed"])
    _, train_loader, val_loader, test_loader = create_dataloaders(
        params["data_dir"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        seed=params["seed"],
    )
    class_names = _class_names(params["data_dir"])
    root_dir = os.path.join(params["output_dir"], "baseline_comparison")
    os.makedirs(root_dir, exist_ok=True)

    results = []
    for index, architecture in enumerate(architectures, start=1):
        # Reset seed để thứ tự model không làm thay đổi phép so sánh.
        set_seed(params["seed"])
        for loader in (train_loader, val_loader, test_loader):
            if loader.generator is not None:
                loader.generator.manual_seed(params["seed"])
        save_dir = os.path.join(root_dir, architecture)
        os.makedirs(save_dir, exist_ok=True)
        logger.info(
            "[%d/%d] Bắt đầu huấn luyện baseline %s",
            index,
            len(architectures),
            architecture,
        )

        started_at = time.time()
        model = ImageClassificationBaseline(
            architecture=architecture,
            num_classes=params["num_classes"],
            pretrained=pretrained,
        )
        train_cfg = {
            "lr_backbone": params["transfer_learning"]["lr_backbone"],
            "lr_head": params["transfer_learning"]["lr_head"],
            "weight_decay": params["weight_decay"],
            "monitor_metric": params.get("monitor_metric", "val_acc"),
            "dvclive_path": os.path.join(save_dir, "dvclive"),
            "save_dir": save_dir,
        }
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            rule_set=None,
            num_epochs=num_epochs,
            patience=params["patience"],
            device=device,
            penalty_weight=0.0,
            num_classes=params["num_classes"],
            config=train_cfg,
        )

        test_accuracy = evaluate_model_performance(
            model,
            test_loader,
            device,
            class_names,
            title=f"Baseline {architecture}",
            output_dir=save_dir,
        )
        plot_training_history(
            history,
            save_dir=save_dir,
            title_suffix=f"Baseline {architecture}",
        )
        result = {
            "architecture": architecture,
            "test_accuracy": test_accuracy,
            "best_val_accuracy": max(history["val_acc"]),
            "epochs_trained": len(history["val_acc"]),
            "elapsed_minutes": (time.time() - started_at) / 60,
            "output_dir": save_dir,
        }
        results.append(result)

        with open(
            os.path.join(root_dir, "summary.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(results, file, ensure_ascii=False, indent=2)

        logger.info(
            "[%d/%d] Hoàn thành %s | test_acc=%.4f",
            index,
            len(architectures),
            architecture,
            test_accuracy,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument(
        "--models",
        "--architectures",
        dest="models",
        nargs="+",
        help="Danh sách baseline cần huấn luyện.",
    )
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    train_all_baselines(args.config, args.models, args.epochs)
