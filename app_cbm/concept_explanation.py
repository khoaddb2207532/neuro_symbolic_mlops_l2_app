# pyright: strict
"""Giải thích CBM thuần Python: ngưỡng riêng từng concept và đóng góp vào logit.

Tách khỏi ``inference.py`` (phụ thuộc torch) để kiểm thử được mà không cần torch.

Với lớp đích ``c``, ``label_predictor`` là Linear nên::

    logit_c = Σ_i W[c, i] * p_i + b_c

Đóng góp của concept ``i`` vào lớp ``c`` là ``p_i * W[c, i]``; tổng đóng góp
cộng bias ``b_c`` bằng đúng logit của ``c``. Đây là phân rã của LOGIT, không
phải của xác suất softmax: đóng góp lớn chỉ cho biết concept đó đẩy logit của
lớp lên hay xuống, không đảm bảo lớp đó thắng các lớp khác.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, cast

DEFAULT_CONCEPT_THRESHOLD = 0.5
THRESHOLDS_KEY = "concept_thresholds"


@dataclass(frozen=True)
class ConceptContribution:
    index: int
    name: str
    probability: float
    threshold: float
    activated: bool
    weight: float
    contribution: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "probability": round(self.probability, 4),
            "threshold": round(self.threshold, 4),
            "activated": self.activated,
            "weight": round(self.weight, 4),
            "contribution": round(self.contribution, 4),
        }


def resolve_concept_thresholds(
    raw: Any,
    num_concepts: int,
    default: float = DEFAULT_CONCEPT_THRESHOLD,
) -> Tuple[List[float], str]:
    """Trả về (ngưỡng của từng concept, nguồn).

    ``raw`` là giá trị ``concept_thresholds`` đọc từ ``cbm_metadata.json``.
    Thiếu khoá thì dùng ``default`` cho mọi concept (nguồn ``"global"``).
    Khoá có mặt nhưng sai định dạng thì báo lỗi, không lặng lẽ bỏ qua,
    vì dùng nhầm ngưỡng sẽ đổi kết quả giải thích mà không ai biết.
    """
    if raw is None:
        return [default] * num_concepts, "global"
    if not isinstance(raw, list):
        raise ValueError(f"{THRESHOLDS_KEY} phải là danh sách.")
    values = cast(List[object], raw)
    if len(values) != num_concepts:
        raise ValueError(
            f"{THRESHOLDS_KEY} phải có đúng {num_concepts} giá trị."
        )
    thresholds: List[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 < value < 1.0
        ):
            raise ValueError(
                f"{THRESHOLDS_KEY} chứa giá trị ngoài khoảng (0, 1): {value!r}."
            )
        thresholds.append(float(value))
    return thresholds, "per_concept"


def concept_contributions(
    probabilities: Sequence[float],
    names: Sequence[str],
    thresholds: Sequence[float],
    weights: Sequence[float],
) -> List[ConceptContribution]:
    """Tính đóng góp ``p_i * W[c, i]`` của từng concept vào lớp đích ``c``.

    ``weights`` là hàng ``W[c, :]`` của lớp đích. Kết quả được xếp theo đóng
    góp giảm dần (concept đẩy logit lên nhiều nhất đứng đầu).
    """
    n = len(probabilities)
    if not (len(names) == len(thresholds) == len(weights) == n):
        raise ValueError(
            "probabilities, names, thresholds và weights phải cùng độ dài."
        )
    rows = [
        ConceptContribution(
            index=index,
            name=names[index],
            probability=float(probabilities[index]),
            threshold=float(thresholds[index]),
            activated=float(probabilities[index]) >= float(thresholds[index]),
            weight=float(weights[index]),
            contribution=float(probabilities[index]) * float(weights[index]),
        )
        for index in range(n)
    ]
    return sorted(rows, key=lambda row: (-row.contribution, row.index))


def logit_residual(
    contributions: Sequence[ConceptContribution],
    bias: float,
    logit: float,
) -> float:
    """Trả về ``|Σ đóng góp + bias − logit|``; về 0 nếu phân rã khớp với mô hình."""
    return abs(sum(row.contribution for row in contributions) + bias - logit)
