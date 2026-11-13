"""Discover and serve baseline, CBM, and rule-regularized CBM checkpoints."""
from __future__ import annotations

import io
import json
import logging
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from PIL import Image, ImageDraw, UnidentifiedImageError

from app.rule_localization import RULE_COLORS, extract_bounding_boxes, image_to_base64
from src.data.dataset import IMAGENET_MEAN, NeuroSymbolicDataset
from src.models.cbm import ConceptBottleneckModel
from src.models.cnn import ImageClassificationBaseline

logger = logging.getLogger(__name__)

MODEL_TYPES = ("baseline", "cbm", "cbm_regularized")
MODEL_LABELS = {
    "baseline": "Baseline",
    "cbm": "CBM",
    "cbm_regularized": "CBM regularized",
}
CONCEPT_ACTIVATION_THRESHOLD = 0.5
MAX_CONCEPT_BOXES = 5


@dataclass
class LoadedModel:
    model_id: str
    model_type: str
    architecture: str
    seed: int
    checkpoint: Path
    model: torch.nn.Module
    class_names: List[str]
    concept_names: List[str]


class CBMInferenceService:
    """Keep all loadable models in memory and refresh when artifacts change."""

    def __init__(
        self,
        artifact_root: Optional[str] = None,
        concept_threshold: float = CONCEPT_ACTIVATION_THRESHOLD,
        occlusion_grid_size: int = 7,
        occlusion_batch_size: int = 16,
    ) -> None:
        self.artifact_root = Path(
            artifact_root if artifact_root is not None else Path(__file__).parent
        ).expanduser().resolve()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.concept_threshold = concept_threshold
        self.occlusion_grid_size = occlusion_grid_size
        self.occlusion_batch_size = occlusion_batch_size
        self.transform = NeuroSymbolicDataset.get_transforms("test")
        self.models: Dict[str, LoadedModel] = {}
        self.statuses: List[Dict[str, Any]] = []
        self._fingerprint: Tuple[Tuple[str, int, int], ...] = ()
        self.refresh(force=True)

    def _run_directories(self) -> List[Path]:
        if not self.artifact_root.is_dir():
            return []
        return sorted(
            (
                path for path in self.artifact_root.glob("outputs_seed_*")
                if path.is_dir() and re.fullmatch(r"outputs_seed_\d+", path.name)
            ),
            key=lambda path: int(path.name.rsplit("_", 1)[1]),
        )

    @staticmethod
    def _checkpoint_for(run_dir: Path, model_type: str) -> Optional[Path]:
        candidates = {
            "baseline": (
                run_dir / "01_baseline" / "baseline_best.pth",
                run_dir / "01_baseline" / "final_model_weights.pth",
            ),
            "cbm": (
                run_dir / "cbm" / "cbm_best.pth",
                run_dir / "cbm" / "final_model_weights.pth",
            ),
            "cbm_regularized": (
                run_dir / "05_rules_model" / "rule_regularized_best.pth",
                run_dir / "05_rules_model" / "final_model_weights.pth",
            ),
        }[model_type]
        return next((path for path in candidates if path.is_file()), None)

    def _discover(self) -> Tuple[List[Dict[str, Any]], Tuple[Tuple[str, int, int], ...]]:
        runs: List[Dict[str, Any]] = []
        tracked_paths = set()
        for run_dir in self._run_directories():
            seed_match = re.search(r"(\d+)$", run_dir.name)
            config_path = run_dir / (
                f"params_seed_{seed_match.group(1)}.yaml" if seed_match else "params.yaml"
            )
            if not config_path.is_file():
                config_path = run_dir / "params.yaml"
            metadata_path = run_dir / "cbm" / "cbm_metadata.json"
            for path in (config_path, metadata_path):
                if path.is_file():
                    tracked_paths.add(path)

            try:
                config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                if not isinstance(config, dict):
                    raise ValueError("params YAML phải chứa một object.")
                seed = int(config.get("seed", seed_match.group(1) if seed_match else 0))
                architecture = (
                    config.get("baseline_comparison", {}).get("selected_architecture")
                    or config.get("architecture")
                )
                if not architecture:
                    raise ValueError("Thiếu baseline_comparison.selected_architecture.")
            except (
                OSError,
                UnicodeDecodeError,
                ValueError,
                TypeError,
                AttributeError,
                yaml.YAMLError,
            ) as exc:
                seed = int(seed_match.group(1)) if seed_match else 0
                architecture = "unknown"
                config_error = f"Không đọc được cấu hình run: {exc}"
            else:
                config_error = None

            metadata: Dict[str, Any] = {}
            metadata_error = None
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if not isinstance(metadata, dict):
                        raise ValueError("metadata phải chứa một JSON object.")
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    metadata_error = f"Không đọc được cbm_metadata.json: {exc}"
                except ValueError as exc:
                    metadata_error = f"cbm_metadata.json không hợp lệ: {exc}"

            run_id = f"{architecture}-seed-{seed}"
            model_statuses = []
            for model_type in MODEL_TYPES:
                checkpoint = self._checkpoint_for(run_dir, model_type)
                if checkpoint is not None:
                    tracked_paths.add(checkpoint)
                model_id = f"{run_id}:{model_type}"
                error = config_error
                if error is None and model_type != "baseline":
                    error = metadata_error
                if checkpoint is None:
                    error = "Không tìm thấy checkpoint trong artifacts của run."
                model_statuses.append(
                    {
                        "model_id": model_id,
                        "model_type": model_type,
                        "model_label": MODEL_LABELS[model_type],
                        "available": False,
                        "checkpoint": str(checkpoint) if checkpoint else None,
                        "error": error,
                        "run_dir": run_dir,
                        "architecture": architecture,
                        "seed": seed,
                        "config": config if not config_error else {},
                        "metadata": metadata,
                    }
                )
            runs.append(
                {
                    "run_id": run_id,
                    "architecture": architecture,
                    "seed": seed,
                    "models": model_statuses,
                }
            )

        fingerprint = tuple(
            sorted(
                (
                    str(path),
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                )
                for path in tracked_paths
            )
        )
        return runs, fingerprint

    @staticmethod
    def _checkpoint_state(checkpoint: Any) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
        if not isinstance(state, dict) or not all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in state.items()
        ):
            raise ValueError("Checkpoint không chứa state_dict hợp lệ.")
        class_names = (
            checkpoint.get("class_names", [])
            if isinstance(checkpoint, dict) else []
        )
        return state, class_names if isinstance(class_names, list) else []

    def _load_model(self, status: Dict[str, Any]) -> LoadedModel:
        checkpoint_path = Path(status["checkpoint"])
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        state, class_names = self._checkpoint_state(checkpoint)
        config = status["config"]
        metadata = status["metadata"]
        model_type = status["model_type"]
        architecture = status["architecture"]
        num_classes = int(config.get("num_classes", len(class_names) or 0))
        if num_classes <= 0:
            raise ValueError("Không xác định được số lớp từ params hoặc checkpoint.")

        concept_names: List[str] = []
        if model_type == "baseline":
            model = ImageClassificationBaseline(
                architecture=architecture, num_classes=num_classes, pretrained=False
            )
        else:
            concept_layer = state.get("concept_layer.0.weight")
            n_concepts = int(
                metadata.get(
                    "n_concepts",
                    concept_layer.shape[0]
                    if concept_layer is not None else config.get("cbm", {}).get("n_concepts", 0),
                )
            )
            if n_concepts <= 0:
                raise ValueError("Không xác định được số concept của checkpoint CBM.")
            concept_names = metadata.get("concept_names") or [
                f"concept_{index}" for index in range(n_concepts)
            ]
            if (
                not isinstance(concept_names, list)
                or len(concept_names) != n_concepts
                or not all(isinstance(name, str) for name in concept_names)
            ):
                raise ValueError(
                    "Danh sách tên concept không khớp với checkpoint."
                )
            architecture = metadata.get("architecture") or architecture
            model = ConceptBottleneckModel(
                architecture=architecture,
                n_concepts=n_concepts,
                num_classes=num_classes,
                pretrained=False,
                concept_names=concept_names,
            )
        model.load_state_dict(state, strict=True)
        model.to(self.device).eval()
        if len(class_names) != num_classes:
            class_names = [f"class_{index}" for index in range(num_classes)]
        return LoadedModel(
            model_id=status["model_id"],
            model_type=model_type,
            architecture=architecture,
            seed=status["seed"],
            checkpoint=checkpoint_path,
            model=model,
            class_names=class_names,
            concept_names=concept_names,
        )

    def refresh(self, force: bool = False) -> None:
        runs, fingerprint = self._discover()
        if not force and fingerprint == self._fingerprint:
            return

        loaded: Dict[str, LoadedModel] = {}
        for run in runs:
            for status in run["models"]:
                if status["checkpoint"] is None or status["error"]:
                    continue
                try:
                    loaded_model = self._load_model(status)
                except (
                    OSError,
                    EOFError,
                    pickle.UnpicklingError,
                    RuntimeError,
                    ValueError,
                    KeyError,
                    TypeError,
                    IndexError,
                    AttributeError,
                ) as exc:
                    status["error"] = f"Không nạp được model: {exc}"
                    logger.exception("Không nạp được model %s", status["model_id"])
                else:
                    status["available"] = True
                    status["error"] = None
                    loaded[loaded_model.model_id] = loaded_model
        self.models = loaded
        self.statuses = runs
        self._fingerprint = fingerprint
        logger.info(
            "CBM inference ready: %d models from %s on %s",
            len(self.models), self.artifact_root, self.device,
        )

    def _concept_occlusion_maps(
        self,
        model: torch.nn.Module,
        image: Image.Image,
        base_concept_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-concept probability drops for each original-image grid cell."""
        width, height = image.size
        grid_size = self.occlusion_grid_size
        mean_color = tuple(round(channel * 255) for channel in IMAGENET_MEAN)
        masked_tensors = []
        for row in range(grid_size):
            for col in range(grid_size):
                x1, x2 = col * width // grid_size, (col + 1) * width // grid_size
                y1, y2 = row * height // grid_size, (row + 1) * height // grid_size
                masked_image = image.copy()
                ImageDraw.Draw(masked_image).rectangle(
                    (x1, y1, x2 - 1, y2 - 1), fill=mean_color
                )
                masked_tensors.append(self.transform(masked_image))

        occluded_concept_probs = []
        for start in range(0, len(masked_tensors), self.occlusion_batch_size):
            batch = torch.stack(
                masked_tensors[start : start + self.occlusion_batch_size]
            ).to(self.device)
            output = model(batch)
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError("CBM không trả concept probabilities.")
            occluded_concept_probs.append(output[1].detach().cpu())

        scores = torch.cat(occluded_concept_probs, dim=0)
        drops = (base_concept_probs.detach().cpu().unsqueeze(0) - scores).clamp_min(0)
        return drops.transpose(0, 1).reshape(-1, grid_size, grid_size)

    def _localize_concepts(
        self,
        model: torch.nn.Module,
        image: Image.Image,
        concept_probs: torch.Tensor,
        concept_names: List[str],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Estimate activated-concept boxes by masking each region of the image."""
        drops = self._concept_occlusion_maps(model, image, concept_probs)
        localized = []
        candidates = []

        for index, probability in enumerate(concept_probs):
            activated = float(probability.item()) >= self.concept_threshold
            boxes = []
            if activated and float(drops[index].max().item()) >= 0.01:
                boxes = extract_bounding_boxes(
                    drops[index],
                    image.size,
                    relative_threshold=0.35,
                    max_regions=2,
                )
                candidates.extend(
                    (float(box["importance"]), index, box) for box in boxes
                )
            localized.append(
                {
                    "name": concept_names[index],
                    "probability": round(float(probability.item()), 4),
                    "activated": activated,
                    "color": None,
                    "boxes": [],
                }
            )

        for _, index, box in sorted(
            candidates, key=lambda candidate: candidate[0], reverse=True
        )[:MAX_CONCEPT_BOXES]:
            color = RULE_COLORS[index % len(RULE_COLORS)]
            localized[index]["color"] = list(color)
            localized[index]["boxes"].append(box)

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        line_width = max(2, min(image.size) // 90)
        for index, concept in enumerate(localized):
            color = concept["color"]
            if color is None:
                continue
            for box in concept["boxes"]:
                overlay_draw.rectangle(
                    (box["x1"], box["y1"], box["x2"], box["y2"]),
                    fill=(*color, 32),
                )

        annotated = Image.alpha_composite(image.convert("RGBA"), overlay)
        draw = ImageDraw.Draw(annotated)
        for index, concept in enumerate(localized):
            color = concept["color"]
            if color is None:
                continue
            for box in concept["boxes"]:
                coordinates = (box["x1"], box["y1"], box["x2"], box["y2"])
                draw.rectangle(coordinates, outline=(*color, 255), width=line_width)
                draw.text(
                    (box["x1"] + 3, box["y1"] + 3),
                    f"C{index + 1}",
                    fill=(*color, 255),
                )
        return localized, image_to_base64(annotated.convert("RGB"))

    def catalog(self) -> Dict[str, Any]:
        self.refresh()
        return {
            "artifact_root": str(self.artifact_root),
            "runs": [
                {
                    "run_id": run["run_id"],
                    "architecture": run["architecture"],
                    "seed": run["seed"],
                    "models": [
                        {
                            key: status[key]
                            for key in (
                                "model_id", "model_type", "model_label",
                                "available", "checkpoint", "error",
                            )
                        }
                        for status in run["models"]
                    ],
                }
                for run in self.statuses
            ],
        }

    def selected_model_ids(
        self,
        run_id: Optional[str] = None,
        model_ids: Optional[List[str]] = None,
    ) -> List[str]:
        self.refresh()
        if model_ids:
            missing = [model_id for model_id in model_ids if model_id not in self.models]
            if missing:
                raise KeyError(f"Model không khả dụng: {', '.join(missing)}")
            selected = model_ids
        elif run_id:
            selected = [
                model_id for model_id in self.models
                if model_id.startswith(f"{run_id}:")
            ]
        else:
            selected = list(self.models)
        if not selected:
            raise LookupError("Không có model khả dụng cho lựa chọn này.")
        return selected

    @torch.inference_mode()
    def predict(
        self,
        image_bytes: bytes,
        model_ids: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        include_localization: bool = True,
    ) -> List[Dict[str, Any]]:
        selected_ids = self.selected_model_ids(run_id=run_id, model_ids=model_ids)
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Nội dung tải lên không phải ảnh hợp lệ.") from exc
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        results = []
        for model_id in selected_ids:
            item = self.models[model_id]
            started = time.perf_counter()
            output = item.model(tensor)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
            predicted_index = int(probabilities.argmax().item())
            top_indices = torch.topk(
                probabilities, min(5, len(item.class_names))
            ).indices.tolist()
            concepts = []
            annotated_image_base64 = None
            if item.model_type != "baseline":
                if not isinstance(output, (tuple, list)) or len(output) < 2:
                    raise RuntimeError(f"{model_id} không trả concept probabilities.")
                concept_probs = output[1].squeeze(0)
                if include_localization:
                    concepts, annotated_image_base64 = self._localize_concepts(
                        item.model, image, concept_probs, item.concept_names
                    )
                else:
                    concepts = [
                        {
                            "name": item.concept_names[index],
                            "probability": round(float(probability.item()), 4),
                            "activated": float(probability.item()) >= self.concept_threshold,
                            "color": None,
                            "boxes": [],
                        }
                        for index, probability in enumerate(concept_probs)
                    ]
                concepts.sort(
                    key=lambda concept: concept["probability"],
                    reverse=True,
                )
            results.append(
                {
                    "model_id": item.model_id,
                    "model_type": item.model_type,
                    "model_label": MODEL_LABELS[item.model_type],
                    "architecture": item.architecture,
                    "seed": item.seed,
                    "predicted_class": item.class_names[predicted_index],
                    "predicted_class_index": predicted_index,
                    "confidence": round(float(probabilities[predicted_index].item()), 4),
                    "top5": [
                        {
                            "label": item.class_names[index],
                            "confidence": round(float(probabilities[index].item()), 4),
                        }
                        for index in top_indices
                    ],
                    "concepts": concepts,
                    "activated_concepts": [
                        concept for concept in concepts if concept["activated"]
                    ],
                    "concept_activation_threshold": (
                        self.concept_threshold if concepts else None
                    ),
                    "annotated_image_base64": annotated_image_base64,
                    "occlusion_method": (
                        "concept_probability_drop" if annotated_image_base64 else None
                    ),
                    "occlusion_grid_size": (
                        self.occlusion_grid_size if annotated_image_base64 else None
                    ),
                    "inference_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            )
        return results
