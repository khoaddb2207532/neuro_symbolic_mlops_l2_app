"""Đánh giá và debug rule set — dùng chung giữa stage4 và các script sweep."""
import torch
from typing import List, Dict
from src.rules.rule_types import Rule


def evaluate_run(
    final_selected_rules: List[Rule],
    valid_rules: List[Rule],
    reward_module,
) -> Dict[str, float]:
    """Tính các thành phần của U(S)/R(S) + vài thống kê mô tả cho MỘT tập
    luật cụ thể, dùng để so sánh khách quan giữa các cấu hình hyperparameter.

    ĐÃ CẬP NHẬT theo `reward.py` mới (per-rule `q_r = freq_r*(1-err_r)*
    exp(-alpha*len_r)`, U(S) = f_quality + lambda_1*f_cover - lambda_2*
    f_overlap - lambda_3*f_size, R(S) = exp(gamma*U(S))):
      - Không còn khái niệm `accuracy`/`coverage`/`conflict_ratio` ở mức
        CẢ TẬP luật (bản cũ tách accuracy trên phần phủ chung + conflict
        chỉ giữa khác target). Thay bằng đúng 4 thành phần mà
        `reward_module.components(s)` trả về: `f_quality` (tổng q_r của
        các luật được chọn), `f_cover` (tỉ lệ union coverage), `f_overlap`
        (tỉ lệ mẫu bị phủ bởi >1 luật, không phân biệt target), `f_size`
        (phần vượt quá K). Dùng chung `components()`/`score()`/`__call__`
        với reward_module để tránh 2 nơi tính công thức khác nhau.
      - `redundancy` (thuần thống kê mô tả, KHÔNG phải thành phần reward):
        trung bình Jaccard theo CẶP luật đã chọn, tính trực tiếp từ
        `reward_module.cover` — bản cũ dùng `reward_module.jaccard` đã
        cache sẵn, nhưng `RuleSetReward` mới không còn giữ ma trận này nên
        phải tính lại tại chỗ (rẻ vì chỉ chạy trên số luật ĐÃ CHỌN, không
        phải toàn bộ n_rules).
      - `score`/`reward`: giá trị U(S) và R(S) thật của tập luật này, tiện
        để so sánh trực tiếp với log/reward quan sát được trong training.
    """
    if not final_selected_rules:
        return {"n_rules": 0, "f_quality": 0.0, "f_cover": 0.0,
                "f_overlap": 0.0, "f_size": 0.0, "redundancy": 0.0,
                "complexity": 0.0, "score": 0.0, "reward": 0.0}

    rule_to_idx = {id(r): i for i, r in enumerate(valid_rules)}
    idx = [rule_to_idx[id(r)] for r in final_selected_rules if id(r) in rule_to_idx]

    device = reward_module.cover.device
    mask = torch.zeros(len(valid_rules), device=device)
    mask[idx] = 1.0
    s = mask.unsqueeze(0)

    n_sel = s.sum(-1)

    comp = reward_module.components(s)
    f_quality = comp["f_quality"].item()
    f_cover = comp["f_cover"].item()
    f_overlap = comp["f_overlap"].item()
    f_size = comp["f_size"].item()

    # `redundancy`: Jaccard trung bình theo CẶP luật đã chọn (toàn bộ overlap,
    # không tách theo target) — thuần mô tả, tính trực tiếp từ cover vì
    # RuleSetReward không còn cache sẵn ma trận jaccard.
    sel_cover = reward_module.cover[idx]                       # (n_sel, n_val)
    n = sel_cover.shape[0]
    if n > 1:
        inter = sel_cover @ sel_cover.T                        # (n_sel, n_sel)
        row_sum = sel_cover.sum(-1, keepdim=True)
        union = (row_sum + row_sum.T - inter).clamp(min=1e-8)
        jaccard = inter / union
        off_diag_sum = jaccard.sum() - jaccard.diagonal().sum()
        redundancy = (off_diag_sum / (n * (n - 1))).item()
    else:
        redundancy = 0.0

    complexity = (n_sel / reward_module.max_rules).item()

    score = reward_module.score(s).item()
    reward = reward_module(s).item()

    return {
        "n_rules": int(n_sel.item()),
        "f_quality": f_quality,
        "f_cover": f_cover,
        "f_overlap": f_overlap,
        "f_size": f_size,
        "redundancy": redundancy,
        "complexity": complexity,
        "score": score,
        "reward": reward,
    }


def debug_breakdown(rule_subset: List[Rule], valid_rules: List[Rule],
                     reward_module, logger, label: str = "") -> None:
    """In log breakdown cho một tập luật — dùng trong training loop để theo dõi."""
    metric = evaluate_run(rule_subset, valid_rules, reward_module)
    logger.info(
        "DEBUG %s: n_selected=%d, f_quality=%.4f, f_cover=%.4f, f_overlap=%.4f, "
        "f_size=%.4f, redundancy=%.4f, complexity=%.4f, score=%.4f, reward=%.4f",
        label, metric["n_rules"], metric["f_quality"], metric["f_cover"],
        metric["f_overlap"], metric["f_size"], metric["redundancy"],
        metric["complexity"], metric["score"], metric["reward"],
    )