"""I/O tiện ích cho luật — dùng chung để tránh lặp code xuất Excel ở nhiều nơi
(trước đây stage3, stage4, và gflownet/pipeline.py mỗi nơi tự viết lại logic
build DataFrame + to_excel giống hệt nhau)."""
from typing import List, Optional

import pandas as pd

from src.rules.rule_types import Rule


def format_rule_with_names(rule: Rule, feature_names: Optional[List[str]] = None) -> str:
    """Format luật với tên concept/feature thực tế thay vì F_i."""
    if not feature_names:
        return str(rule)
    cond_strs = []
    for cond in rule.conditions:
        name = (
            feature_names[cond.feature_index]
            if 0 <= cond.feature_index < len(feature_names)
            else f"F_{cond.feature_index}"
        )
        cond_strs.append(f"{name} {cond.operator} {cond.threshold:.3f}")
    return " AND ".join(cond_strs) + f" => Class {rule.target_class}"


def save_rules_excel(
    rules: List[Rule],
    path: str,
    feature_names: Optional[List[str]] = None,
) -> None:
    """Lưu danh sách luật ra file Excel với các cột chuẩn hoá dùng chung
    xuyên suốt pipeline (stage3 luật hợp lệ, stage4 luật được GFlowNet chọn)."""
    rows = [
        {
            "STT": idx,
            "Target Class": rule.target_class,
            "Confidence (%)": round(rule.confidence * 100, 2),
            "#Features": len(rule.conditions),
            "Rule": format_rule_with_names(rule, feature_names),
        }
        for idx, rule in enumerate(rules, 1)
    ]
    pd.DataFrame(rows).to_excel(path, index=False)
