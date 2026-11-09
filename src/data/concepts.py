"""Dataset wrapper cung cấp concept annotations cho CBM.

Concept annotations là vector nhị phân [n_concepts] cho mỗi ảnh, đánh dấu
concept nào xuất hiện trong ảnh đó. Có 2 chế độ cung cấp annotations:

1. **File annotation** (khuyến nghị): đọc từ file JSON (VLM labeling hoặc manual).
   Hỗ trợ cả 2 định dạng phổ biến:
   - Định dạng VLM/Kaggle (`image_concepts` với dict `{"class": ..., "vector": [...]}`)
   - Định dạng flat (`annotations` với key: `[0, 1, ...]`)

2. **Attribute-based** (tự động): dựa trên class label, tạo concept annotations
   từ ma trận class-to-concepts (class i có những concept nào).

Lưu ý:
  Số lượng concept cần chọn cẩn thận:
  - Quá ít: mất thông tin, giảm acc.
  - Quá nhiều (đặc biệt nếu concept chung chung như "mây", "nước"): loãng
    concept, mô hình không phân biệt được các lớp → interpretability giảm.
  - Concept tốt: ĐẶC TRƯNG cho lớp này mà lớp khác KHÔNG có.
"""
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.data.dataset import NeuroSymbolicDataset
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ConceptAnnotatedDataset(Dataset):
    """Dataset trả về (image, label, concept_vector) cho CBM training.

    Args:
        base_dataset: Dataset gốc (NeuroSymbolicDataset).
        concept_annotations: Dict mapping image_path (hoặc filename/relative path) → concept_vector.
        class_to_concepts: Dict mapping class_idx → concept_vector.
            Dùng làm fallback khi một ảnh chưa có nhãn concept cụ thể.
        n_concepts: Số lượng concepts.
    """

    def __init__(
        self,
        base_dataset: NeuroSymbolicDataset,
        concept_annotations: Optional[Dict[str, list]] = None,
        class_to_concepts: Optional[Dict[int, list]] = None,
        n_concepts: int = 30,
    ):
        self.base_dataset = base_dataset
        self.n_concepts = n_concepts
        self.concept_annotations = concept_annotations
        self.class_to_concepts = class_to_concepts

        if concept_annotations is None and class_to_concepts is None:
            raise ValueError(
                "Cần cung cấp ít nhất một trong: concept_annotations "
                "(per-image) hoặc class_to_concepts (per-class)."
            )

    @property
    def classes(self):
        return self.base_dataset.classes

    @property
    def class_to_idx(self):
        return self.base_dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        image, label = self.base_dataset[idx]
        sample_path = self.base_dataset.samples[idx][0]

        concept_vec = None
        if self.concept_annotations is not None:
            # Tra cứu đa tầng (multi-key lookup) để khớp path linh hoạt
            raw_str = str(sample_path)
            norm_str = raw_str.replace("\\", "/")
            fname = sample_path.name
            class_rel = f"{sample_path.parent.name}/{fname}"
            split_rel = f"{getattr(self.base_dataset, 'split', '')}/{class_rel}".strip("/")

            for candidate in (raw_str, norm_str, split_rel, class_rel, fname):
                if candidate in self.concept_annotations:
                    concept_vec = self.concept_annotations[candidate]
                    break

        if concept_vec is None:
            # Fallback 1: Dùng vector prototype của class nếu có
            if self.class_to_concepts is not None and label in self.class_to_concepts:
                concept_vec = self.class_to_concepts[label]
            else:
                concept_vec = [0.0] * self.n_concepts

        concept_tensor = torch.tensor(concept_vec, dtype=torch.float32)
        return image, label, concept_tensor


def load_concept_annotations(
    annotations_path: str,
    class_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, list], List[str], int, Optional[Dict[int, list]]]:
    """Load concept annotations từ file JSON.

    Hỗ trợ 2 định dạng:
    1. Định dạng VLM/Kaggle (`image_concepts`):
       {
           "concept_names": [...],
           "class_names": {"0": "class_a", ...},
           "image_concepts": {
               "split/class/img.jpg": {"class": "class_a", "vector": [0, 1, ...]}
           }
       }
    2. Định dạng flat (`annotations`):
       {
           "concept_names": [...],
           "annotations": {"path/to/img.jpg": [0, 1, ...]}
       }

    Đồng thời tự động tính vector concept trung bình cho từng class làm fallback.

    Returns:
        (annotations_dict, concept_names, n_concepts, class_to_concepts_fallback)
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    concept_names = data.get("concept_names", [])
    n_concepts = len(concept_names)

    raw_items = data.get("image_concepts") or data.get("annotations") or {}
    annotations: Dict[str, list] = {}
    class_vectors_collector: Dict[str, List[list]] = {}

    for k, v in raw_items.items():
        if isinstance(v, dict) and "vector" in v:
            vec = v["vector"]
            cls_name = v.get("class")
            if cls_name:
                class_vectors_collector.setdefault(str(cls_name), []).append(vec)
        else:
            vec = v

        norm_k = k.replace("\\", "/").strip("/")
        annotations[k] = vec
        annotations[norm_k] = vec
        annotations[Path(k).name] = vec
        parts = norm_k.split("/")
        if len(parts) >= 2:
            annotations["/".join(parts[-2:])] = vec

    # Tự động tính vector concept fallback cho từng class
    class_to_concepts: Optional[Dict[int, list]] = None
    json_class_names = data.get("class_names")

    # Xây dựng class_name -> class_idx mapping
    name_to_idx = {}
    if class_names:
        name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    elif json_class_names and isinstance(json_class_names, dict):
        name_to_idx = {v: int(k) for k, v in json_class_names.items()}

    if class_vectors_collector and name_to_idx:
        class_to_concepts = {}
        for cls_name, vecs in class_vectors_collector.items():
            if cls_name in name_to_idx and vecs:
                idx = name_to_idx[cls_name]
                # Lấy trung bình rồi làm tròn (hoặc binarize > 0.3)
                mean_vec = np.mean(vecs, axis=0)
                class_to_concepts[idx] = (mean_vec > 0.3).astype(float).tolist()

    logger.info(
        "Đã load %d concepts, %d mẫu ảnh từ %s",
        n_concepts, len(raw_items), annotations_path,
    )
    if class_to_concepts:
        logger.info("Đã tự động trích xuất prototype concept cho %d classes.", len(class_to_concepts))

    return annotations, concept_names, n_concepts, class_to_concepts


def build_class_to_concepts(
    class_concept_matrix_path: str,
) -> Tuple[Dict[int, list], List[str], int]:
    """Load ma trận class → concepts từ file JSON."""
    with open(class_concept_matrix_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    concept_names = data["concept_names"]
    n_concepts = len(concept_names)
    class_to_concepts = {int(k): v for k, v in data["class_concepts"].items()}
    return class_to_concepts, concept_names, n_concepts


def generate_default_class_concepts(
    class_names: List[str],
    n_concepts: int = 30,
    output_path: Optional[str] = None,
) -> Tuple[Dict[int, list], List[str]]:
    """Tạo ma trận class-concept MẪU (placeholder) khi chưa có file gán nhãn."""
    num_classes = len(class_names)
    concepts_per_class = max(2, n_concepts // num_classes)
    concept_names = []

    for cls_name in class_names:
        for j in range(concepts_per_class):
            concept_names.append(f"{cls_name}_đặctrưng_{j}")
            if len(concept_names) >= n_concepts - 3:
                break
        if len(concept_names) >= n_concepts - 3:
            break

    shared_concepts = ["màu_sắc_nổi_bật", "cấu_trúc_hình_học", "nền_tự_nhiên"]
    for sc in shared_concepts:
        if len(concept_names) < n_concepts:
            concept_names.append(sc)

    while len(concept_names) < n_concepts:
        concept_names.append(f"concept_{len(concept_names)}")
    concept_names = concept_names[:n_concepts]

    class_to_concepts = {}
    for cls_idx, cls_name in enumerate(class_names):
        vec = [0] * n_concepts
        start_idx = cls_idx * concepts_per_class
        for j in range(concepts_per_class):
            idx = start_idx + j
            if idx < n_concepts:
                vec[idx] = 1
        for shared_idx in range(n_concepts - 3, n_concepts):
            if shared_idx < n_concepts:
                vec[shared_idx] = 1 if (cls_idx % 2 == 0) else 0
        class_to_concepts[cls_idx] = vec

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "concept_names": concept_names,
                    "class_concepts": {str(k): v for k, v in class_to_concepts.items()},
                    "_note": "Template tự sinh.",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return class_to_concepts, concept_names
