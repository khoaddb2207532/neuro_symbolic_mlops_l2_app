"""Dataset wrapper cung cấp concept annotations cho CBM.

Concept annotations là vector nhị phân [n_concepts] cho mỗi ảnh, đánh dấu
concept nào xuất hiện trong ảnh đó.  Có 2 chế độ cung cấp annotations:

1. **File annotation** (khuyến nghị): đọc từ file CSV/JSON do người dùng gán.
   Mỗi dòng chứa: image_path, concept_0, concept_1, ..., concept_N (0 hoặc 1).

2. **Attribute-based** (tự động): dựa trên class label, tạo concept annotations
   từ ma trận class-to-concepts (class i có những concept nào).  Cách này
   KHÔNG chính xác bằng file annotation (vì một ảnh thuộc class "chùa" không
   nhất thiết có "rồng"), nhưng là baseline chấp nhận được khi chưa có nhãn
   concept thủ công.

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


class ConceptAnnotatedDataset(Dataset):
    """Dataset trả về (image, label, concept_vector) cho CBM training.

    Args:
        base_dataset: Dataset gốc (NeuroSymbolicDataset).
        concept_annotations: Dict mapping image_path → concept_vector (list/ndarray).
            Nếu None, dùng class_to_concepts để tạo tự động.
        class_to_concepts: Dict mapping class_idx → concept_vector.
            Dùng khi không có concept annotations per-image.
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

        if self.concept_annotations is not None:
            img_path = str(self.base_dataset.samples[idx][0])
            concept_vec = self.concept_annotations.get(img_path)
            if concept_vec is None:
                # Fallback to class-based
                if self.class_to_concepts is not None:
                    concept_vec = self.class_to_concepts[label]
                else:
                    concept_vec = [0.0] * self.n_concepts
        else:
            concept_vec = self.class_to_concepts[label]

        concept_tensor = torch.tensor(concept_vec, dtype=torch.float32)
        return image, label, concept_tensor


def load_concept_annotations(
    annotations_path: str,
) -> Tuple[Dict[str, list], List[str], int]:
    """Load concept annotations từ file JSON.

    File JSON format::

        {
            "concept_names": ["nón_lá", "áo_dài", "sông", "thuyền", ...],
            "annotations": {
                "path/to/img1.jpg": [1, 0, 1, 0, ...],
                "path/to/img2.jpg": [0, 1, 0, 1, ...],
                ...
            }
        }

    Returns:
        (annotations_dict, concept_names, n_concepts)
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    concept_names = data["concept_names"]
    n_concepts = len(concept_names)
    annotations = {k: v for k, v in data["annotations"].items()}
    return annotations, concept_names, n_concepts


def build_class_to_concepts(
    class_concept_matrix_path: str,
) -> Tuple[Dict[int, list], List[str], int]:
    """Load ma trận class → concepts từ file JSON.

    File JSON format::

        {
            "concept_names": ["nón_lá", "áo_dài", "sông", ...],
            "class_concepts": {
                "0": [1, 0, 1, 0, ...],
                "1": [0, 1, 0, 1, ...],
                ...
            }
        }

    Mỗi class có một binary vector chỉ ra NHỮNG concept ĐẶC TRƯNG cho class
    đó.  Ví dụ class "áo_dài" có concept "áo_dài"=1, "vải"=1, "nón_lá"=0.

    Returns:
        (class_to_concepts_dict, concept_names, n_concepts)
    """
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
    """Tạo ma trận class-concept MẪU (placeholder) để bắt đầu nhanh.

    Mỗi class được gán ~3-5 concept "riêng" (concept_i = 1 CHỈ cho class đó)
    cộng thêm vài concept chung.  Đây là TEMPLATE, người dùng cần chỉnh sửa
    cho phù hợp với domain thực tế.

    Returns:
        (class_to_concepts, concept_names)
    """
    num_classes = len(class_names)

    # Tạo concept names: mỗi class có ~2 concept riêng + vài concept chung
    concepts_per_class = max(2, n_concepts // num_classes)
    concept_names = []

    # Concept riêng cho từng class
    for cls_name in class_names:
        for j in range(concepts_per_class):
            concept_names.append(f"{cls_name}_đặctrưng_{j}")
            if len(concept_names) >= n_concepts - 3:
                break
        if len(concept_names) >= n_concepts - 3:
            break

    # Thêm concept chung (shared)
    shared_concepts = ["màu_sắc_nổi_bật", "cấu_trúc_hình_học", "nền_tự_nhiên"]
    for sc in shared_concepts:
        if len(concept_names) < n_concepts:
            concept_names.append(sc)

    # Pad nếu thiếu
    while len(concept_names) < n_concepts:
        concept_names.append(f"concept_{len(concept_names)}")
    concept_names = concept_names[:n_concepts]

    # Tạo matrix: mỗi class bật concept riêng + concept chung
    class_to_concepts = {}
    for cls_idx, cls_name in enumerate(class_names):
        vec = [0] * n_concepts
        # Bật concept riêng
        start_idx = cls_idx * concepts_per_class
        for j in range(concepts_per_class):
            idx = start_idx + j
            if idx < n_concepts:
                vec[idx] = 1
        # Bật concept chung (50% chance)
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
                    "_note": (
                        "Đây là template tự sinh. Hãy chỉnh sửa concept_names "
                        "và class_concepts cho phù hợp với domain thực tế. "
                        "Concept tốt: ĐẶC TRƯNG cho 1-2 class, KHÔNG xuất hiện "
                        "ở tất cả các class (tránh loãng)."
                    ),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return class_to_concepts, concept_names
