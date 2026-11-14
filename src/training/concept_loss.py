# pyright: strict
"""Concept loss cho CBM: BCE có logits, pos_weight theo concept, bỏ qua nhãn ``ignore``.

Quy ước nhãn concept (một vector ``[n_concepts]`` mỗi ảnh):
  - giá trị trong [0, 1]: nhãn (0/1, hoặc mềm).
  - NaN: ``ignore`` — vị trí này không tham gia loss (không biết concept có hay không).

``pos_weight[c] = #âm_c / #dương_c`` (đã kẹp trong [1, max]) bù mất cân bằng: nhiều concept
hiếm, nếu không có trọng số thì mô hình học dự đoán "không có" cho gần như mọi concept.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def compute_pos_weight(targets: np.ndarray, max_weight: float = 10.0) -> np.ndarray:
    """Tính pos_weight từ ma trận nhãn ``[N, C]`` (NaN = ignore).

    Concept chưa từng xuất hiện (``#dương = 0``) nhận trọng số 1 để không bùng nổ.
    """
    if targets.ndim != 2:
        raise ValueError("targets phải có dạng [N, n_concepts].")
    if not math.isfinite(max_weight) or max_weight < 1.0:
        raise ValueError("max_weight phải là số hữu hạn >= 1.")
    valid = ~np.isnan(targets)
    positives = np.where(valid, targets, 0.0).sum(axis=0)
    observed = valid.sum(axis=0).astype(np.float64)
    negatives = observed - positives
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(positives > 0, negatives / np.maximum(positives, 1e-12), 1.0)
    return np.clip(raw, 1.0, max_weight).astype(np.float32)


def masked_concept_bce_with_logits(
    concept_logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCE có logits, trung bình trên các vị trí KHÔNG phải ``ignore``.

    Vị trí NaN không đóng góp vào tổng và không đóng góp vào mẫu số, nên việc
    thiếu nhãn không làm loss bị kéo về 0. Nếu toàn bộ batch là ``ignore``, trả về 0
    (có gradient bằng 0) thay vì NaN.
    """
    if concept_logits.shape != targets.shape:
        raise ValueError(
            f"Kích thước logits {tuple(concept_logits.shape)} khác targets {tuple(targets.shape)}."
        )
    mask = torch.isfinite(targets)
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    elementwise = F.binary_cross_entropy_with_logits(
        concept_logits, safe_targets, pos_weight=pos_weight, reduction="none"
    )
    masked = torch.where(mask, elementwise, torch.zeros_like(elementwise))
    denominator = mask.sum().clamp_min(1).to(masked.dtype)
    return masked.sum() / denominator
