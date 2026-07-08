"""Đánh giá và debug rule set — dùng chung giữa stage4 và các script sweep."""
import torch
from typing import List, Dict
from src.rules.rule_types import Rule


def evaluate_run(
    final_selected_rules: List[Rule],
    valid_rules: List[Rule],
    reward_module,
) -> Dict[str, float]:
    """Tính accuracy/coverage/redundancy/complexity cho MỘT tập luật cụ thể,
    dùng để so sánh khách quan giữa các cấu hình hyperparameter.

    LƯU Ý: `redundancy`/`complexity` ở đây là THỐNG KÊ MÔ TẢ (để báo cáo,
    theo dõi tính gọn/dễ đọc của tập luật) — KHÔNG còn là thành phần của hàm
    reward mà GFlowNet tối ưu (xem reward.py: score() giờ chỉ còn
    accuracy + coverage - w_conflict * conflict_ratio).
      - `redundancy` (tính trên self.jaccard đầy đủ, trung bình theo CẶP
        luật) đo TOÀN BỘ trùng lặp bất kể target — thuần thống kê mô tả.
      - `redundancy_conflict` giờ lấy TRỰC TIẾP từ
        `reward_module.components(s)["conflict_ratio"]` — đúng con số thật
        mà GFlowNet đang bị phạt (tỉ lệ MẪU bị phủ bởi >=2 target khác nhau
        trong số luật đã chọn, KHÔNG chia theo số cặp luật — xem docstring
        trong reward.py để biết vì sao đổi cách đo này). Dùng chung 1 hàm
        `components()` với `score()` để tránh 2 nơi tính công thức khác
        nhau (rủi ro cũ: evaluation.py tự tính lại bằng công thức pairwise
        đã lỗi thời, gây lệch số liệu báo cáo so với reward thật).
    """
    if not final_selected_rules:
        return {"n_rules": 0, "accuracy": 0.0, "coverage": 0.0,
                "redundancy": 0.0, "redundancy_conflict": 0.0,
                "complexity": 0.0, "f1_like": 0.0}

    rule_to_idx = {id(r): i for i, r in enumerate(valid_rules)}
    idx = [rule_to_idx[id(r)] for r in final_selected_rules if id(r) in rule_to_idx]

    device = reward_module.cover.device
    mask = torch.zeros(len(valid_rules), device=device)
    mask[idx] = 1.0
    s = mask.unsqueeze(0)

    n_sel = s.sum(-1)

    comp = reward_module.components(s)
    accuracy = comp["accuracy"].item()
    coverage = comp["coverage"].item()
    redundancy_conflict = comp["conflict_ratio"].item()

    # `redundancy` (toàn bộ overlap, không tách theo target) vẫn tính riêng
    # từ self.jaccard đầy đủ — thuần mô tả, không liên quan tới việc
    # GFlowNet có bị phạt vì nó hay không.
    n_pairs = (n_sel * (n_sel - 1)).clamp(min=1)
    pair_red = (s.unsqueeze(1) * s.unsqueeze(2) * reward_module.jaccard).sum((-1, -2))
    redundancy = (pair_red / n_pairs).item()

    complexity = (n_sel / reward_module.max_rules).item()

    f1_like = 2 * accuracy * coverage / (accuracy + coverage + 1e-8)

    return {
        "n_rules": int(n_sel.item()),
        "accuracy": accuracy,
        "coverage": coverage,
        "redundancy": redundancy,
        "redundancy_conflict": redundancy_conflict,
        "complexity": complexity,
        "f1_like": f1_like,
    }


def debug_breakdown(rule_subset: List[Rule], valid_rules: List[Rule],
                     reward_module, logger, label: str = "") -> None:
    """In log breakdown cho một tập luật — dùng trong training loop để theo dõi."""
    metric = evaluate_run(rule_subset, valid_rules, reward_module)
    logger.info(
        "DEBUG %s: n_selected=%d, accuracy=%.4f, coverage=%.4f, "
        "redundancy=%.4f, redundancy_conflict=%.4f, complexity=%.4f, f1_like=%.4f",
        label, metric["n_rules"], metric["accuracy"], metric["coverage"],
        metric["redundancy"], metric["redundancy_conflict"], metric["complexity"], metric["f1_like"],
    )