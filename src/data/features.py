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
import torch
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from src.rules.extractor import RuleExtractor
from src.rules.rule_types import RuleSet
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def extract_and_save_features(model, dataloaders: dict, output_dir: str, device="cuda") -> None:
    model = model.to(device).eval()
    os.makedirs(output_dir, exist_ok=True)
    valid_splits = [s for s in ["train", "val", "test"] if s in dataloaders and dataloaders[s] is not None]
    with torch.no_grad():
        for split in valid_splits:
            all_features, all_labels = [], []
            for images, labels in tqdm(dataloaders[split], desc=f"Extracting {split}"):
                feats = model(images.to(device))
                all_features.append(feats.cpu())
                all_labels.append(labels.cpu())
            torch.save(torch.cat(all_features, 0), os.path.join(output_dir, f"{split}_features.pt"))
            torch.save(torch.cat(all_labels, 0), os.path.join(output_dir, f"{split}_labels.pt"))
            logger.info("Saved %s: %d-dim features", split, all_features[0].shape[-1])


def train_and_save_rf(
    features_dir: str,
    rf_output_path: str,
    rules_output_path: str,
    n_estimators: int = 100,
) -> RuleSet:
    """Train RF một lần trên toàn bộ train set, lưu model, đồng thời trích
    luật thô (chưa lọc) ngay tại đây và lưu ra `rules_output_path` — nơi khác
    trong pipeline (stage3) chỉ cần gọi lại luật đã trích, không train/trích
    lại lần nữa.
    """
    X_train = torch.load(f"{features_dir}/train_features.pt").numpy()
    y_train = torch.load(f"{features_dir}/train_labels.pt").numpy()

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, rf_output_path)
    logger.info("RF model saved to %s", rf_output_path)

    raw_rules = RuleExtractor().extract(rf)
    logger.info("Số luật thô trích từ RF: %d", len(raw_rules))
    with open(rules_output_path, "wb") as f:
        pickle.dump(raw_rules, f)
    logger.info("Luật thô saved to %s", rules_output_path)

    return raw_rules
