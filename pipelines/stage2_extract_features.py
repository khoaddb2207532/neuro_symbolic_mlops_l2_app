"""DVC Stage 2 — Trích đặc trưng từ model đã chọn (Baseline CNN hoặc CBM).

Hỗ trợ cả 2 chế độ:
  1. Baseline CNN (mặc định): trích latent features (1280-d, 576-d, ...)
  2. CBM (ngữ nghĩa concept): trích concept_probs (n_concepts-d trong [0, 1])
     kèm theo lưu tên concept để Stage 3 & 4 hiển thị luật người đọc được.

Chạy:
  python -m pipelines.stage2_extract_features --config params.yaml
  python -m pipelines.stage2_extract_features --config params.yaml --source cbm
"""
import argparse
import json
import os

import torch

from src.data.dataset import create_dataloaders
from src.data.features import extract_and_save_features
from src.models.cbm import ConceptBottleneckModel
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


def main(params_path: str, source_override: str = None) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloaders, _, _, _ = create_dataloaders(
        params["data_dir"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        seed=params["seed"],
        include_val2=True,
    )

    source = source_override or params.get("features", {}).get("source", "baseline")
    output_dir = os.path.join(params["output_dir"], "02_features")
    os.makedirs(output_dir, exist_ok=True)

    if source == "cbm":
        # Trích đặc trưng từ CBM (concept probabilities)
        cbm_dir = os.path.join(params["output_dir"], "cbm")
        metadata_path = os.path.join(cbm_dir, "cbm_metadata.json")
        concept_names = None
        n_concepts = params.get("cbm", {}).get("n_concepts", 30)
        arch = selected_baseline_architecture(params)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            concept_names = meta.get("concept_names")
            n_concepts = meta.get("n_concepts", n_concepts)
            arch = meta.get("architecture", arch)

        cbm_ckpt = os.path.join(cbm_dir, "cbm_best.pth")
        if not os.path.exists(cbm_ckpt):
            cbm_ckpt = os.path.join(cbm_dir, "final_model_weights.pth")

        feature_extractor = ConceptBottleneckModel(
            architecture=arch,
            n_concepts=n_concepts,
            num_classes=params["num_classes"],
            pretrained=False,
            concept_names=concept_names,
        )
        load_model_weights(feature_extractor, cbm_ckpt, device, required=True)
        logger.info(
            "Trích đặc trưng Concept từ CBM (%d concepts, backbone=%s) tại %s",
            n_concepts, arch, cbm_ckpt,
        )

        # Lưu danh sách tên concept để Stage 3 & 4 dùng
        if concept_names:
            names_path = os.path.join(output_dir, "feature_names.json")
            with open(names_path, "w", encoding="utf-8") as f:
                json.dump(concept_names, f, ensure_ascii=False, indent=2)
            logger.info("Đã lưu %d tên concept vào %s", len(concept_names), names_path)

    else:
        # Chế độ cũ: trích latent features từ Baseline CNN
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
        logger.info("Trích đặc trưng latent từ Baseline CNN '%s'", architecture)

    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    extract_and_save_features(feature_extractor, dataloaders, output_dir=output_dir, device=device)
    logger.info("Stage 2 hoàn thành. Đặc trưng lưu tại %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument(
        "--source",
        choices=["baseline", "cbm"],
        default=None,
        help="Nguồn trích feature: 'baseline' (latent) hoặc 'cbm' (concept probabilities)",
    )
    args = parser.parse_args()
    main(args.config, source_override=args.source)
