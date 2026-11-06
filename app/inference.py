"""Inference for the baseline and two Stage-5 model products."""
from __future__ import annotations

import io
import os
import pickle
import time
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image

from app.rule_localization import (
    RULE_COLORS,
    draw_rule_boxes,
    extract_bounding_boxes,
    image_to_base64,
    model_tensor_to_pil,
)
from src.data.dataset import NeuroSymbolicDataset
from src.models.cnn import ImageClassificationBaseline
from src.rules.penalty import BinaryTransformer
from src.rules.rule_types import RuleSet
from src.utils.checkpoint import load_model_weights
from src.utils.config import (
    load_params,
    selected_baseline_architecture,
    selected_baseline_checkpoint,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


MODEL_ORDER = ("baseline", "stage5", "stage5_bayesian")


class InferenceService:
    """Load all available products once and compare them on the same input."""

    def __init__(
        self,
        params_path: str = "params.yaml",
        top_k_rules: int = 3,
        localization_grid_size: int = 7,
        localization_batch_size: int = 16,
    ):
        self.params = load_params(params_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k_rules = top_k_rules
        self.localization_grid_size = localization_grid_size
        self.localization_batch_size = localization_batch_size
        self.class_names = self._load_class_names()
        self.transformer = BinaryTransformer(temperature=10.0)
        self.transform = NeuroSymbolicDataset.get_transforms("test")
        self.variants = self._load_variants()

        available = [key for key in MODEL_ORDER if self.variants[key]["available"]]
        if not available:
            expected = ", ".join(
                self.variants[key]["expected_checkpoint"] for key in MODEL_ORDER
            )
            raise FileNotFoundError(f"Không tìm thấy checkpoint phục vụ nào: {expected}")
        self.primary_key = "stage5" if "stage5" in available else available[0]

        logger.info(
            "InferenceService sẵn sàng | device=%s | models=%s",
            self.device,
            ",".join(available),
        )

    @property
    def model(self):
        """Backward-compatible primary model used by the existing API."""
        return self.variants[self.primary_key]["model"]

    @property
    def rule_set(self) -> RuleSet:
        """Backward-compatible primary rule set used by the health endpoint."""
        return self.variants[self.primary_key]["rule_set"]

    def _load_class_names(self) -> List[str]:
        try:
            dataset = NeuroSymbolicDataset(self.params["data_dir"], "test")
            return [
                name
                for name, _ in sorted(
                    dataset.class_to_idx.items(), key=lambda item: item[1]
                )
            ]
        except FileNotFoundError:
            logger.warning("Không tìm thấy data_dir; dùng tên lớp theo chỉ số.")
            return [f"class_{index}" for index in range(self.params["num_classes"])]

    @staticmethod
    def _first_existing_path(candidates: Sequence[str]) -> Optional[str]:
        return next((path for path in candidates if os.path.exists(path)), None)

    def _variant_specs(self) -> List[Dict]:
        output_dir = self.params["output_dir"]
        baseline_candidates = [
            selected_baseline_checkpoint(self.params),
            os.path.join(output_dir, "01_baseline", "final_model_weights.pth"),
            os.path.join(output_dir, "01_baseline", "baseline_best.pth"),
        ]
        return [
            {
                "key": "baseline",
                "label": "Baseline CNN",
                "short_label": "Baseline",
                "description": "CNN thuần, không regularize bằng luật.",
                "checkpoint_candidates": baseline_candidates,
                "rules_path": None,
                "train_command": "dvc repro train_baseline",
            },
            {
                "key": "stage5",
                "label": "Điều chuẩn theo luật",
                "short_label": "Theo luật",
                "description": "CNN tinh chỉnh với tập luật GFlowNet cố định.",
                "checkpoint_candidates": [
                    os.path.join(
                        output_dir, "05_rules_model", "final_model_weights.pth"
                    )
                ],
                "rules_path": os.path.join(
                    output_dir, "04_filtered_rules", "selected_rules.pkl"
                ),
                "train_command": "dvc repro train_rule_regularized",
            },
            {
                "key": "stage5_bayesian",
                "label": "Tổng hợp đa tập luật",
                "short_label": "Đa tập luật",
                "description": "CNN được điều chuẩn bằng bằng chứng tổng hợp từ nhiều tập luật GFlowNet.",
                "checkpoint_candidates": [
                    os.path.join(
                        output_dir,
                        "05b_rules_model_bayesian",
                        "final_model_weights.pth",
                    )
                ],
                "rules_path": os.path.join(
                    output_dir,
                    "05b_rules_model_bayesian",
                    "gflownet_sampled_rules.pkl",
                ),
                "train_command": "dvc repro train_rule_regularized_bayesian",
            },
        ]

    def _load_model(self, model_path: str) -> ImageClassificationBaseline:
        model = ImageClassificationBaseline(
            architecture=selected_baseline_architecture(self.params),
            num_classes=self.params["num_classes"],
            pretrained=False,
        )
        load_model_weights(model, model_path, self.device, required=True)
        model.to(self.device).eval()
        return model

    @staticmethod
    def _load_rules(rules_path: Optional[str]) -> RuleSet:
        if not rules_path or not os.path.exists(rules_path):
            return RuleSet(rules=[])
        with open(rules_path, "rb") as file:
            rules = pickle.load(file)
        return rules if isinstance(rules, RuleSet) else RuleSet(rules=rules)

    def _load_variants(self) -> Dict[str, Dict]:
        variants: Dict[str, Dict] = {}
        for spec in self._variant_specs():
            checkpoint = self._first_existing_path(spec["checkpoint_candidates"])
            variant = {
                **spec,
                "expected_checkpoint": spec["checkpoint_candidates"][0],
                "checkpoint": checkpoint,
                "model": None,
                "rule_set": RuleSet(rules=[]),
                "available": False,
                "error": None,
            }
            if checkpoint is None:
                variant["error"] = "Chưa có checkpoint."
            else:
                try:
                    variant["model"] = self._load_model(checkpoint)
                    variant["rule_set"] = self._load_rules(spec["rules_path"])
                    variant["available"] = True
                except Exception as exc:  # noqa: BLE001
                    variant["error"] = f"Không nạp được checkpoint: {exc}"
                    logger.exception("Không nạp được model %s", spec["key"])
            variants[spec["key"]] = variant
        return variants

    def model_statuses(self) -> List[Dict]:
        statuses = []
        for key in MODEL_ORDER:
            variant = self.variants[key]
            statuses.append(
                {
                    "model_key": key,
                    "model_label": variant["label"],
                    "available": variant["available"],
                    "checkpoint": variant["checkpoint"],
                    "num_rules": len(variant["rule_set"]),
                    "error": variant["error"],
                    "train_command": variant["train_command"],
                }
            )
        return statuses

    def _localize_rules(self, model, tensor, selected_rules, base_scores):
        """Estimate spatial rule evidence by batched mean-color occlusion."""
        visible_image = model_tensor_to_pil(tensor.squeeze(0))
        if not selected_rules:
            return [], visible_image

        _, _, height, width = tensor.shape
        grid = self.localization_grid_size
        occluded = tensor.repeat(grid * grid, 1, 1, 1)
        for row in range(grid):
            for col in range(grid):
                index = row * grid + col
                y1, y2 = row * height // grid, (row + 1) * height // grid
                x1, x2 = col * width // grid, (col + 1) * width // grid
                occluded[index, :, y1:y2, x1:x2] = 0

        feature_chunks = []
        for start in range(0, len(occluded), self.localization_batch_size):
            _, features = model(occluded[start : start + self.localization_batch_size])
            feature_chunks.append(features)
        occluded_features = torch.cat(feature_chunks, dim=0)
        local_rule_set = RuleSet(rules=list(selected_rules))
        occluded_scores = self.transformer.transform(occluded_features, local_rule_set)
        base = torch.tensor(base_scores, device=occluded_scores.device).unsqueeze(0)
        importance = (base - occluded_scores).clamp_min(0)

        rule_regions = []
        for rule_index, (rule, score) in enumerate(zip(selected_rules, base_scores)):
            boxes = extract_bounding_boxes(
                importance[:, rule_index].reshape(grid, grid), visible_image.size
            )
            if not boxes:
                continue
            color = RULE_COLORS[rule_index % len(RULE_COLORS)]
            rule_regions.append(
                {
                    "rule_number": rule_index + 1,
                    "rule": str(rule),
                    "satisfaction_score": round(score, 4),
                    "color": list(color),
                    "boxes": boxes,
                    "method": "rule_aware_occlusion",
                }
            )
        return rule_regions, draw_rule_boxes(visible_image, rule_regions)

    def _predict_variant(self, variant: Dict, tensor: torch.Tensor) -> Dict:
        started = time.perf_counter()
        model = variant["model"]
        logits, features = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        prediction_ms = (time.perf_counter() - started) * 1000

        matched_rules = []
        selected_rules = []
        selected_scores = []
        rule_set = variant["rule_set"]
        if len(rule_set) > 0:
            satisfaction = self.transformer.transform(features, rule_set).squeeze(0)
            satisfied_indices = torch.where(satisfaction > 0.5)[0].tolist()
            candidates = [
                (rule_set.rules[index], float(satisfaction[index].item()))
                for index in satisfied_indices
                if rule_set.rules[index].target_class == pred_idx
            ]
            candidates.sort(key=lambda item: item[1], reverse=True)
            for rule, score in candidates[: self.top_k_rules]:
                selected_rules.append(rule)
                selected_scores.append(score)
                matched_rules.append(
                    {
                        "rule": str(rule),
                        "satisfaction_score": round(score, 4),
                        "rule_confidence": round(rule.confidence, 4),
                    }
                )

        rule_regions, annotated_image = self._localize_rules(
            model, tensor, selected_rules, selected_scores
        )
        total_ms = (time.perf_counter() - started) * 1000
        return {
            "model_key": variant["key"],
            "model_label": variant["label"],
            "model_description": variant["description"],
            "available": True,
            "predicted_class": self.class_names[pred_idx],
            "predicted_class_index": pred_idx,
            "confidence": round(float(probs[pred_idx].item()), 4),
            "top5": [
                {
                    "class": self.class_names[index],
                    "confidence": round(float(probs[index].item()), 4),
                }
                for index in torch.topk(
                    probs, min(5, len(self.class_names))
                ).indices.tolist()
            ],
            "matched_rules": matched_rules,
            "rule_regions": rule_regions,
            "annotated_image_base64": image_to_base64(annotated_image),
            "inference_ms": round(prediction_ms, 1),
            "explanation_ms": round(max(0.0, total_ms - prediction_ms), 1),
        }

    @torch.no_grad()
    def compare(self, image_bytes: bytes) -> Dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        results = []
        for key in MODEL_ORDER:
            variant = self.variants[key]
            if variant["available"]:
                results.append(self._predict_variant(variant, tensor))
            else:
                results.append(
                    {
                        "model_key": key,
                        "model_label": variant["label"],
                        "model_description": variant["description"],
                        "available": False,
                        "error": variant["error"],
                        "expected_checkpoint": variant["expected_checkpoint"],
                        "train_command": variant["train_command"],
                    }
                )
        return {"models": results}

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> Dict:
        """Keep /predict compatible by serving the fixed-rule model when available."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return self._predict_variant(self.variants[self.primary_key], tensor)
