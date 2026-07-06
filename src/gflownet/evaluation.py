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
    dùng để so sánh khách quan giữa các cấu hình hyperparameter."""
    if not final_selected_rules:
        return {"n_rules": 0, "accuracy": 0.0, "coverage": 0.0,
                "redundancy": 0.0, "complexity": 0.0, "f1_like": 0.0}

    rule_to_idx = {id(r): i for i, r in enumerate(valid_rules)}
    idx = [rule_to_idx[id(r)] for r in final_selected_rules if id(r) in rule_to_idx]

    device = reward_module.cover.device
    mask = torch.zeros(len(valid_rules), device=device)
    mask[idx] = 1.0
    s = mask.unsqueeze(0)

    n_sel = s.sum(-1)
    covered = (s @ reward_module.cover) > 0
    correct_cov = (s @ reward_module.correct) > 0
    accuracy = (correct_cov.float().sum(-1) / covered.float().sum(-1).clamp(min=1)).item()
    coverage = covered.float().mean(-1).item()

    pair_red = (s.unsqueeze(1) * s.unsqueeze(2) * reward_module.jaccard).sum((-1, -2))
    n_pairs = (n_sel * (n_sel - 1)).clamp(min=1)
    redundancy = (pair_red / n_pairs).item()
    complexity = (n_sel / reward_module.max_rules).item()

    f1_like = 2 * accuracy * coverage / (accuracy + coverage + 1e-8)

    return {
        "n_rules": int(n_sel.item()),
        "accuracy": accuracy,
        "coverage": coverage,
        "redundancy": redundancy,
        "complexity": complexity,
        "f1_like": f1_like,
    }


def debug_breakdown(rule_subset: List[Rule], valid_rules: List[Rule],
                     reward_module, logger, label: str = "") -> None:
    """In log breakdown cho một tập luật — dùng trong training loop để theo dõi."""
    metric = evaluate_run(rule_subset, valid_rules, reward_module)
    logger.info(
        "DEBUG %s: n_selected=%d, accuracy=%.4f, coverage=%.4f, "
        "redundancy=%.4f, complexity=%.4f, f1_like=%.4f",
        label, metric["n_rules"], metric["accuracy"], metric["coverage"],
        metric["redundancy"], metric["complexity"], metric["f1_like"],
    )