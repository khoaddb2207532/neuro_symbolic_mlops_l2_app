"""DVC Stage 2 — Trích đặc trưng từ baseline đã chọn."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import torch

from src.data.dataset import create_dataloaders
from src.data.features import extract_and_save_features
from src.models.cnn import build_selected_baseline, selected_baseline_checkpoint
from src.utils.checkpoint import load_model_weights
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main(params_path: str) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloaders, _, _, _ = create_dataloaders(
        params["data_dir"], batch_size=params["batch_size"], num_workers=params["num_workers"],
        seed=params["seed"], drop_last_train=False,
    )
    trained_model_path = selected_baseline_checkpoint(params)
    if not os.path.exists(trained_model_path):
        raise FileNotFoundError(f"CNN checkpoint for feature extraction is missing: {trained_model_path}")
    feature_extractor = build_selected_baseline(params, pretrained=False)
    load_model_weights(feature_extractor, trained_model_path, device, required=True)
    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False
    output_dir = os.path.join(params["output_dir"], "02_features")
    extract_and_save_features(feature_extractor, dataloaders, output_dir=output_dir,
                              device=device, contract_dir="data")
    digest = hashlib.sha256()
    with open(trained_model_path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    with open(os.path.join("data", "features_metadata.json"), "w", encoding="utf-8") as stream:
        json.dump({"architecture": feature_extractor.architecture,
                   "feature_dim": feature_extractor.feature_dim,
                   "cnn_checkpoint": os.path.abspath(trained_model_path),
                   "cnn_checkpoint_sha256": digest.hexdigest(),
                   "generated_at": datetime.now(timezone.utc).isoformat(),
                   "train_samples": len(dataloaders["train"].dataset),
                   "val_samples": len(dataloaders["val"].dataset)}, stream, indent=2)
    logger.info("Stage 2 hoàn thành. Đặc trưng lưu tại %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
