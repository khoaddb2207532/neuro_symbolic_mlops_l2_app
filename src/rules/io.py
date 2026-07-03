"""I/O tiện ích cho luật — dùng chung để tránh lặp code xuất Excel ở nhiều nơi
(trước đây stage3, stage4, và gflownet/pipeline.py mỗi nơi tự viết lại logic
build DataFrame + to_excel giống hệt nhau)."""
from typing import List

import pandas as pd

from src.rules.rule_types import Rule


def save_rules_excel(rules: List[Rule], path: str) -> None:
    """Lưu danh sách luật ra file Excel với các cột chuẩn hoá dùng chung
    xuyên suốt pipeline (stage3 luật hợp lệ, stage4 luật được GFlowNet chọn)."""
    rows = [
        {
            "STT": idx,
            "Target Class": rule.target_class,
            "Confidence (%)": round(rule.confidence * 100, 2),
            "#Features": len(rule.conditions),
            "Rule": str(rule),
        }
        for idx, rule in enumerate(rules, 1)
    ]
    pd.DataFrame(rows).to_excel(path, index=False)
