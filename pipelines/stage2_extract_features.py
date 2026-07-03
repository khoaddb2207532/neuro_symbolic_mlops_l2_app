"""DVC Stage 2 — Trích đặc trưng 1280-d từ CNN đã fine-tune."""
import argparse
import os

import torch

from src.data.dataset import create_dataloaders
from src.data.features import extract_and_save_features
from src.models.cnn import FeatureExtractor
from src.utils.config import load_params
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
    trained_model_path = os.path.join(params["output_dir"], "01_baseline", "baseline_best.pth")
    feature_extractor = FeatureExtractor(
        num_classes=params["num_classes"], trained_model_path=trained_model_path, device=device
    )
    output_dir = os.path.join(params["output_dir"], "02_features")
    extract_and_save_features(feature_extractor, dataloaders, output_dir=output_dir, device=device)
    logger.info("Stage 2 hoàn thành. Đặc trưng lưu tại %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
