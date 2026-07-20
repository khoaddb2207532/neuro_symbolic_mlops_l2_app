"""Evaluate the regularized CNN checkpoint on validation and test splits.

Run:
    python -m pipelines.evaluate_regularized_cnn --config params.yaml
"""
import argparse
import json
import os

import torch

from src.data.dataset import NeuroSymbolicDataset, create_dataloaders
from src.evaluation.evaluate import evaluate_classification_metrics, evaluate_model_performance
from src.models.cnn import CNNBaseline
from src.utils.checkpoint import load_model_weights
from src.utils.config import load_params
from src.utils.seed import set_seed


def main(config_path: str, checkpoint_path: str, output_dir: str) -> None:
    params = load_params(config_path)
    set_seed(params["seed"])
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"regularized CNN checkpoint is missing: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, val_loader, test_loader = create_dataloaders(
        params["data_dir"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        seed=params["seed"],
        drop_last_train=False,
    )
    test_dataset = NeuroSymbolicDataset(params["data_dir"], "test")
    class_names = [name for name, _ in sorted(test_dataset.class_to_idx.items(),
                                               key=lambda item: item[1])]
    if len(class_names) != params["num_classes"]:
        raise ValueError(
            f"test split has {len(class_names)} classes, expected {params['num_classes']}"
        )

    model = CNNBaseline(params["num_classes"], freeze_stage="last_block")
    load_model_weights(model, checkpoint_path, device, required=True)
    model = model.to(device).eval()
    os.makedirs(output_dir, exist_ok=True)

    validation_metrics = evaluate_classification_metrics(model, val_loader, device)
    test_metrics = evaluate_classification_metrics(model, test_loader, device)
    evaluate_model_performance(
        model, val_loader, device, class_names,
        title="Regularized CNN Validation", output_dir=output_dir,
    )
    evaluate_model_performance(
        model, test_loader, device, class_names,
        title="Regularized CNN Test", output_dir=output_dir,
    )

    result = {
        "checkpoint": checkpoint_path,
        "device": str(device),
        "validation": validation_metrics,
        "test": test_metrics,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Evaluation artifacts: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/cnn_regularized.pt")
    parser.add_argument("--output-dir", default="reports/regularized_evaluation")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output_dir)
