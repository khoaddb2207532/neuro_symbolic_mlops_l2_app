"""Trích và lưu đặc trưng CNN; huấn luyện Random Forest trên đặc trưng đó + trích
luật thô ngay tại chỗ.

`train_and_save_rf()` là nơi DUY NHẤT train RandomForestClassifier trong toàn
bộ pipeline. Trước đây `GPUFastRuleValidator.validate_crossval()` còn tự train
thêm K RandomForest khác trên các fold của train — trùng lặp trách nhiệm và
không có gì đảm bảo tổng quát hoá tốt hơn val set thật. Đã bỏ (xem
src/rules/validator.py); việc lọc luật giờ dùng val set thật qua
`GPUFastRuleValidator.validate()`.
"""
import os
import pickle

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from src.rules.extractor import RuleExtractor
from src.rules.rule_types import RuleSet
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def extract_and_save_features(model, dataloaders: dict, output_dir: str, device="cuda",
                              contract_dir: str = "data") -> None:
    """Trích và lưu CẢ features 1280-d LẪN logits (num_classes-d), cùng một
    lần forward — không forward lại CNN ở stage4 để tính uncertainty nữa.

    `model` là `FeatureExtractor` (forward() chỉ trả features, dừng ở
    classifier[0:3]) — logits được tính thêm bằng đúng lớp Linear cuối cùng
    của CÙNG backbone (`model.backbone.classifier[3]`), không đổi
    `FeatureExtractor.forward()` để tránh ảnh hưởng các nơi khác đang gọi nó
    chỉ để lấy features (vd stage3 trích luật)."""
    model = model.to(device).eval()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(contract_dir, exist_ok=True)
    valid_splits = [s for s in ["train", "val", "test"] if s in dataloaders and dataloaders[s] is not None]
    with torch.no_grad():
        for split in valid_splits:
            all_features, all_logits, all_labels = [], [], []
            for images, labels in tqdm(dataloaders[split], desc=f"Extracting {split}"):
                logits, feats = model(images.to(device))
                all_features.append(feats.cpu())
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
            features = torch.cat(all_features, 0)
            logits = torch.cat(all_logits, 0)
            labels = torch.cat(all_labels, 0)
            if features.shape[0] != labels.shape[0]:
                raise ValueError(f"{split}: feature/label count mismatch")
            expected_count = len(dataloaders[split].dataset)
            if features.shape[0] != expected_count:
                raise ValueError(
                    f"{split}: extracted {features.shape[0]} features for {expected_count} dataset samples"
                )
            if not torch.isfinite(features).all():
                raise ValueError(f"{split}: penultimate features contain NaN or infinity")
            torch.save(features, os.path.join(output_dir, f"{split}_features.pt"))
            torch.save(logits, os.path.join(output_dir, f"{split}_logits.pt"))
            torch.save(labels, os.path.join(output_dir, f"{split}_labels.pt"))
            if split in ("train", "val"):
                np.save(os.path.join(contract_dir, f"features_{split}.npy"), features.numpy())
                np.save(os.path.join(contract_dir, f"labels_{split}.npy"), labels.numpy())
            if split == "val":
                np.save(os.path.join(contract_dir, "cnn_predictions_val.npy"),
                        logits.argmax(dim=1).numpy())
            logger.info(
                "Saved %s: %d-dim features, %d-class logits",
                split, all_features[0].shape[-1], all_logits[0].shape[-1],
            )


def train_and_save_rf(
    features_dir: str,
    rf_output_path: str,
    rules_output_path: str,
    n_estimators: int = 150,
    max_depth: int = 13,
) -> RuleSet:
    """Train RF một lần trên toàn bộ train set, lưu model, đồng thời trích
    luật thô (chưa lọc) ngay tại đây và lưu ra `rules_output_path` — nơi khác
    trong pipeline (stage3) chỉ cần gọi lại luật đã trích, không train/trích
    lại lần nữa.
    """
    X_train = torch.load(f"{features_dir}/train_features.pt").numpy()
    y_train = torch.load(f"{features_dir}/train_labels.pt").numpy()

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, rf_output_path)
    logger.info("RF model saved to %s", rf_output_path)

    raw_rules = RuleExtractor().extract(rf)
    # logger.info("Số luật thô trích từ RF: %d", len(raw_rules))
    # with open(rules_output_path, "wb") as f:
    #     pickle.dump(raw_rules, f)
    # logger.info("Luật thô saved to %s", rules_output_path)

    return raw_rules
