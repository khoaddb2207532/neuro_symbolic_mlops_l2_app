"""DVC Stage 2 — Trích đặc trưng từ baseline đã chọn."""
import argparse
import os

import torch

from src.data.dataset import create_dataloaders
from src.data.features import extract_and_save_features
from src.models.cnn import ImageClassificationBaseline
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

    dataloaders, _, _, _ = create_dataloaders(
        params["data_dir"], batch_size=params["batch_size"], num_workers=params["num_workers"], seed=params["seed"]
    )
    architecture = selected_baseline_architecture(params)
    trained_model_path = selected_baseline_checkpoint(params)
    feature_extractor = ImageClassificationBaseline(
        architecture=architecture,
        num_classes=params["num_classes"],
        pretrained=False,
    )
    load_model_weights(
        feature_extractor, trained_model_path, device, required=True
    )
    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False
    output_dir = os.path.join(params["output_dir"], "02_features")
    extract_and_save_features(feature_extractor, dataloaders, output_dir=output_dir, device=device)
    logger.info(
        "Stage 2 hoàn thành với baseline '%s'. Đặc trưng lưu tại %s",
        architecture,
        output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
