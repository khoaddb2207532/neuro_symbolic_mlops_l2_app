"""So sánh trực tiếp chất lượng các tập luật trên validation set.

Tất cả phương pháp được đánh giá bằng đúng ``RuleSetReward`` và các tensor
cover/correct/rule_len/labels đã được Stage 4 lưu trong
``gflownet_rule_order.pkl``. Không trích feature hoặc validate luật lại.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from src.gflownet.reward import RuleSetReward
from src.gflownet.rule_ranking_analysis import load_rule_order
from src.rules.rule_types import Rule
from src.utils.config import load_params


METHOD_PATHS = {
    "gflownet_elite": "04_filtered_rules",
    "random": "04_filtered_rules_random",
    "topk_confidence": "04_filtered_rules_topk_confidence",
    "greedy_coverage": "04_filtered_rules_greedy_coverage",
}


def _condition_signature(condition) -> Tuple[int, str, float]:
    return (
        int(condition.feature_index),
        str(condition.operator),
        round(float(condition.threshold), 10),
    )


def _rule_signature(rule: Rule) -> Tuple:
    return (
        tuple(_condition_signature(condition) for condition in rule.conditions),
        int(rule.target_class),
    )


def _load_selected_rules(path: Path) -> List[Rule]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as file:
        rules = pickle.load(file)
    if hasattr(rules, "rules"):
        rules = rules.rules
    return list(rules)


def _indices_in_rule_order(
    selected_rules: Sequence[Rule],
    ordered_rules: Sequence[Rule],
) -> List[int]:
    # Dùng queue index cho mỗi signature để xử lý an toàn cả luật trùng.
    available: Dict[Tuple, List[int]] = {}
    for index, rule in enumerate(ordered_rules):
        available.setdefault(_rule_signature(rule), []).append(index)

    selected_indices = []
    for rule in selected_rules:
        signature = _rule_signature(rule)
        candidates = available.get(signature, [])
        if not candidates:
            raise ValueError(
                "Luật đã chọn không tồn tại trong gflownet_rule_order.pkl: "
                f"{rule}"
            )
        selected_indices.append(candidates.pop(0))
    return selected_indices


def _mean_pairwise_jaccard(
    reward: RuleSetReward,
    indices: Sequence[int],
) -> float:
    if len(indices) < 2:
        return 0.0
    index_tensor = torch.tensor(indices, device=reward.jaccard.device)
    matrix = reward.jaccard.index_select(0, index_tensor).index_select(
        1, index_tensor
    )
    upper = torch.triu_indices(
        len(indices),
        len(indices),
        offset=1,
        device=matrix.device,
    )
    return float(matrix[upper[0], upper[1]].mean().item())


def _sample_overlap_ratio(
    cover: torch.Tensor,
    indices: Sequence[int],
) -> float:
    if not indices:
        return 0.0
    selected_cover = cover[list(indices)].float()
    return float((selected_cover.sum(dim=0) > 1).float().mean().item())


def evaluate_method(
    method: str,
    selected_rules: Sequence[Rule],
    selected_indices: Sequence[int],
    reward: RuleSetReward,
    cover: torch.Tensor,
    rule_len: torch.Tensor,
) -> Dict:
    state = torch.zeros(
        1,
        cover.shape[0],
        dtype=torch.float32,
        device=cover.device,
    )
    state[0, list(selected_indices)] = 1.0
    components = reward.components(state)
    raw_score = reward.score(state)
    positive_reward = reward(state)
    lengths = rule_len[list(selected_indices)].float()
    confidences = torch.tensor(
        [float(rule.confidence) for rule in selected_rules],
        device=cover.device,
    )

    return {
        "method": method,
        "n_rules_selected": len(selected_indices),
        "reward_score": float(raw_score.item()),
        "positive_reward": float(positive_reward.item()),
        "macro_accuracy": float(components["macro_accuracy"].item()),
        "coverage": float(components["coverage"].item()),
        "correct_coverage": float(components["correct_coverage"].item()),
        "wrong_coverage": float(components["wrong_coverage"].item()),
        "conflict_ratio": float(components["conflict_ratio"].item()),
        "mean_pairwise_jaccard": _mean_pairwise_jaccard(
            reward, selected_indices
        ),
        "sample_overlap_ratio": _sample_overlap_ratio(
            cover, selected_indices
        ),
        "mean_rule_length": float(lengths.mean().item()),
        "mean_rule_confidence": float(confidences.mean().item()),
    }


def _write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: str) -> Tuple[Path, Path]:
    params = load_params(config_path)
    output_dir = Path(params["output_dir"])
    filtered_dir = output_dir / "04_filtered_rules"
    rule_order = load_rule_order(str(filtered_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ordered_rules = list(rule_order["valid_rules"])
    cover = rule_order["cover"].to(device)
    correct = rule_order["correct"].to(device)
    rule_len = rule_order["rule_len"].to(device)
    labels = rule_order["labels"].to(device)
    targets = torch.tensor(
        [int(rule.target_class) for rule in ordered_rules],
        dtype=torch.long,
        device=device,
    )
    confidences = torch.tensor(
        [float(rule.confidence) for rule in ordered_rules],
        dtype=torch.float32,
        device=device,
    )

    reward = RuleSetReward(
        cover=cover,
        correct=correct,
        rule_len=rule_len,
        max_rules=int(rule_order["max_rules"]),
        targets=targets,
        labels=labels,
        confidences=confidences,
        w_acc=float(rule_order.get("w_acc", 1.0)),
        w_cov=float(rule_order.get("w_cov", 0.5)),
        w_wrong=float(rule_order.get("w_wrong", 0.75)),
        w_conflict=float(rule_order.get("w_conflict", 0.1)),
        beta=float(rule_order.get("beta", 3.0)),
    )

    rows = []
    expected_budget = None
    for method, directory_name in METHOD_PATHS.items():
        selected_rules = _load_selected_rules(
            output_dir / directory_name / "selected_rules.pkl"
        )
        indices = _indices_in_rule_order(selected_rules, ordered_rules)
        if expected_budget is None:
            expected_budget = len(indices)
        elif len(indices) != expected_budget:
            raise RuntimeError(
                f"Vi phạm matched-budget: {method}={len(indices)}, "
                f"GFlowNet={expected_budget}."
            )
        row = evaluate_method(
            method,
            selected_rules,
            indices,
            reward,
            cover,
            rule_len,
        )
        row = {
            "seed": int(params["seed"]),
            "backbone": params["baseline_comparison"][
                "selected_architecture"
            ],
            "loss_type": rule_order["loss_type"],
            **row,
        }
        rows.append(row)

    csv_path = output_dir / "rule_set_quality_comparison.csv"
    json_path = output_dir / "rule_set_quality_comparison.json"
    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nCHẤT LƯỢNG TẬP LUẬT TRÊN VALIDATION SET")
    print(
        f"{'Method':<20} {'Reward':>9} {'MacroAcc':>9} "
        f"{'Coverage':>9} {'Correct':>9} {'Wrong':>9} "
        f"{'Conflict':>9} {'Jaccard':>9} {'AvgLen':>8}"
    )
    for row in rows:
        print(
            f"{row['method']:<20} "
            f"{row['reward_score']:>9.4f} "
            f"{row['macro_accuracy']:>9.4f} "
            f"{row['coverage']:>9.4f} "
            f"{row['correct_coverage']:>9.4f} "
            f"{row['wrong_coverage']:>9.4f} "
            f"{row['conflict_ratio']:>9.4f} "
            f"{row['mean_pairwise_jaccard']:>9.4f} "
            f"{row['mean_rule_length']:>8.2f}"
        )
    print(" -", csv_path)
    print(" -", json_path)
    return csv_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(arguments.config)
