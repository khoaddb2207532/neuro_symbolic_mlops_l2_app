"""Đánh giá các baseline đã huấn luyện trên toàn bộ tập train.

Script chỉ nạp ``outputs/01_baselines/<model>/model.pt`` và chạy inference,
không thực hiện huấn luyện lại.

Ví dụ:
    python -m pipelines.evaluate_baselines_train --config params.yaml
    python -m pipelines.evaluate_baselines_train --config params.yaml --models resnet50 vit
"""
import argparse
import json
import os
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from src.data.dataset import NeuroSymbolicDataset
from src.evaluation.evaluate import (
    evaluate_classification_metrics,
    evaluate_model_performance,
)
from src.models.cnn import VisionBaseline, canonical_baseline_name
from src.utils.checkpoint import load_model_weights
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import seed_worker, set_seed

logger = get_logger(__name__)


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def _build_train_eval_loader(params: dict) -> tuple[DataLoader, list[str]]:
    # Dùng transform xác định giống val/test thay vì augmentation ngẫu nhiên
    # của quá trình huấn luyện; đồng thời đánh giá đủ mọi mẫu.
    dataset = NeuroSymbolicDataset(
        params["data_dir"],
        "train",
        transform=NeuroSymbolicDataset.get_transforms("val"),
    )
    generator = torch.Generator()
    generator.manual_seed(params["seed"])
    loader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=params["num_workers"],
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=seed_worker,
    )
    class_names = [
        name for name, _ in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])
    ]
    return loader, class_names


def evaluate_train(params_path: str, requested_models: Iterable[str] | None = None) -> dict:
    params = load_params(params_path)
    baseline_cfg = params.get("baselines", {})
    raw_models = list(requested_models or baseline_cfg.get("models", []))
    if not raw_models:
        raise ValueError("Configure baselines.models or pass --models.")

    model_names = [canonical_baseline_name(name) for name in raw_models]
    if len(model_names) != len(set(model_names)):
        raise ValueError("The baseline list contains duplicate models (possibly via aliases).")

    set_seed(params["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, class_names = _build_train_eval_loader(params)
    root_dir = os.path.join(params["output_dir"], "01_baselines")
    all_results = {}

    for name in model_names:
        model_dir = os.path.join(root_dir, name)
        checkpoint_path = os.path.join(model_dir, "model.pt")
        logger.info("Đánh giá %s trên tập train từ %s", name, checkpoint_path)

        # pretrained=False vì checkpoint sẽ ghi đè toàn bộ trọng số và không cần tải
        # weights ImageNet từ Internet.
        model = VisionBaseline(name, params["num_classes"], pretrained=False).to(device)
        load_model_weights(model, checkpoint_path, device, required=True)

        train_metrics = evaluate_classification_metrics(model, train_loader, device)
        title = f"{name.replace('_', ' ').title()} Train"
        evaluate_model_performance(
            model, train_loader, device, class_names, title, model_dir
        )

        train_result_path = os.path.join(model_dir, "train_metrics.json")
        _write_json(train_result_path, train_metrics)

        # Bổ sung kết quả train nhưng giữ nguyên validation/test đã có.
        metrics_path = os.path.join(model_dir, "metrics.json")
        combined_metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as stream:
                combined_metrics = json.load(stream)
        combined_metrics["train"] = train_metrics
        _write_json(metrics_path, combined_metrics)

        all_results[name] = train_metrics
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = os.path.join(root_dir, "train_evaluation_summary.json")
    _write_json(summary_path, all_results)
    logger.info("Đã lưu tổng hợp đánh giá train tại %s", summary_path)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Các baseline cần đánh giá; mặc định lấy toàn bộ baselines.models.",
    )
    args = parser.parse_args()
    evaluate_train(args.config, args.models)
