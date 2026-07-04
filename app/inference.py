"""Lớp inference: load model + rule set 1 lần khi service khởi động, tái sử
dụng cho mọi request. Tách khỏi src/ vì đây là logic phục vụ (serving), không
phải logic huấn luyện — giữ ranh giới rõ giữa training pipeline và serving app.
"""
import io
import os
import pickle
from typing import Dict, List

import torch
from PIL import Image

from src.data.dataset import NeuroSymbolicDataset
from src.models.cnn import CNNBaseline
from src.rules.penalty import BinaryTransformer
from src.rules.rule_types import RuleSet
from src.utils.checkpoint import load_model_weights
from src.utils.config import load_params
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class InferenceService:
    def __init__(self, params_path: str = "params.yaml", top_k_rules: int = 5):
        self.params = load_params(params_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k_rules = top_k_rules

        out_dir = self.params["output_dir"]
        model_path = os.path.join(out_dir, "05_rules_model", "final_model_weights.pth")
        rules_path = os.path.join(out_dir, "04_filtered_rules", "selected_rules.pkl")

        self.class_names = self._load_class_names()
        self.model = self._load_model(model_path)
        self.rule_set = self._load_rules(rules_path)
        self.transformer = BinaryTransformer(temperature=10.0)
        self.transform = NeuroSymbolicDataset.get_transforms("test")

        logger.info(
            "InferenceService sẵn sàng | device=%s | classes=%d | rules=%d",
            self.device, len(self.class_names), len(self.rule_set),
        )

    def _load_class_names(self) -> List[str]:
        try:
            ds = NeuroSymbolicDataset(self.params["data_dir"], "test")
            return [n for n, _ in sorted(ds.class_to_idx.items(), key=lambda x: x[1])]
        except FileNotFoundError:
            logger.warning("Không tìm thấy data_dir để lấy tên class, dùng chỉ số thay thế.")
            return [f"class_{i}" for i in range(self.params["num_classes"])]

    def _load_model(self, model_path: str) -> CNNBaseline:
        model = CNNBaseline(num_classes=self.params["num_classes"], freeze_stage="full")
        # required=True: app serving PHẢI có model đã huấn luyện, không được
        # âm thầm chạy trên trọng số ImageNet chưa fine-tune.
        load_model_weights(model, model_path, self.device, required=True)
        model.to(self.device).eval()
        return model

    def _load_rules(self, rules_path: str) -> RuleSet:
        if not os.path.exists(rules_path):
            logger.warning("Không có file luật tại %s — service vẫn chạy nhưng không giải thích được.", rules_path)
            return RuleSet(rules=[])
        with open(rules_path, "rb") as f:
            rules = pickle.load(f)
        return rules if isinstance(rules, RuleSet) else RuleSet(rules=rules)

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> Dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits, features = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())

        matched_rules = []
        if len(self.rule_set) > 0:
            binary = self.transformer.transform(features, self.rule_set).squeeze(0)  # (n_rules,)
            satisfied_idx = torch.where(binary > 0.5)[0].tolist()
            candidates = [
                (self.rule_set.rules[i], float(binary[i].item()))
                for i in satisfied_idx
                if self.rule_set.rules[i].target_class == pred_idx
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)
            for rule, score in candidates[: self.top_k_rules]:
                matched_rules.append(
                    {
                        "rule": str(rule),
                        "satisfaction_score": round(score, 4),
                        "rule_confidence": round(rule.confidence, 4),
                    }
                )

        return {
            "predicted_class": self.class_names[pred_idx],
            "predicted_class_index": pred_idx,
            "confidence": round(float(probs[pred_idx].item()), 4),
            "top5": [
                {"class": self.class_names[i], "confidence": round(float(probs[i].item()), 4)}
                for i in torch.topk(probs, min(5, len(self.class_names))).indices.tolist()
            ],
            "matched_rules": matched_rules,
        }
