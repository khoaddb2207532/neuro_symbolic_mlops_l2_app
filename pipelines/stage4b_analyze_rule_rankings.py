"""DVC Stage 4b — Phân tích SAU KHI GFlowNet đã train xong (KHÔNG train lại
bất kỳ tham số nào, chỉ nạp checkpoint sampler để sample).

  1. Sample K trajectory từ policy đã train.
  2. p_include(i) = tần suất luật i xuất hiện trong K tập được sample.
  3. So sánh ranking theo p_include với 2 ranking "ngây thơ":
       - topk_confidence      (rule.confidence giảm dần)
       - marginal_gain_alone  (điểm reward của luật đứng MỘT MÌNH — gần
                               giống bước đầu tiên của greedy_coverage)

Output:
  outputs/04_filtered_rules/rule_ranking_analysis.csv
  outputs/04_filtered_rules/rule_ranking_analysis_summary.txt
"""
import argparse
import csv
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr

from src.gflownet.rule_ranking_analysis import (
    load_rule_order,
    rank_topk_confidence,
    ranking_from_scores,
    rebuild_gflownet,
    sample_inclusion_probabilities,
    score_single_rules,
    topk_overlap,
)
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

def main(params_path: str = "params.yaml", K: int = None) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if K is None:
        K = params.get("rule_ranking_analysis", {}).get("K", 300)

    output_dir = os.path.join(params["output_dir"], "04_filtered_rules")
    checkpoint_name = params.get("rule_ranking_analysis", {}).get(
        "checkpoint", "gflownet_best_diverse.pth"
    )
    ckpt_path = os.path.join(output_dir, checkpoint_name)
    if not os.path.exists(ckpt_path):
        fallback = os.path.join(
            output_dir, "gflownet_best_converged.pth"
        )
        logger.warning(
            "Không có %s — fallback sang %s.",
            checkpoint_name,
            os.path.basename(fallback),
        )
        ckpt_path = fallback

    rule_order = load_rule_order(output_dir)
    gflownet, env, valid_rules, reward_module = rebuild_gflownet(rule_order, ckpt_path, device)
    n_rules = len(valid_rules)
    logger.info("Nạp lại policy đã train: %d luật, loss_type=%s", n_rules, rule_order["loss_type"])

    if n_rules == 0:
        logger.warning("Không có luật hợp lệ nào — bỏ qua phân tích ranking.")
        return

    logger.info("Sampling K=%d trajectory từ policy đã train (KHÔNG train lại)...", K)
    p_include = sample_inclusion_probabilities(gflownet, env, n_rules, K=K)

    rank_p = ranking_from_scores(p_include)
    rank_conf = rank_topk_confidence(valid_rules)
    singleton_scores = score_single_rules(
        reward_module, n_rules, device
    )
    rank_alone = ranking_from_scores(singleton_scores)

    # Correlation trực tiếp trên score, không correlation giữa ordinal rank
    # tự tạo. scipy xử lý tie đúng cách; Kendall mặc định là tau-b.
    p_numpy = p_include.numpy()
    confidence_numpy = np.asarray(
        [float(rule.confidence) for rule in valid_rules]
    )
    singleton_numpy = singleton_scores.numpy()
    spearman_conf = float(
        spearmanr(p_numpy, confidence_numpy).statistic
    )
    spearman_alone = float(
        spearmanr(p_numpy, singleton_numpy).statistic
    )
    kendall_conf = float(
        kendalltau(p_numpy, confidence_numpy, variant="b").statistic
    )
    kendall_alone = float(
        kendalltau(p_numpy, singleton_numpy, variant="b").statistic
    )

    # Top-k dùng đúng số luật elite GFlowNet thực sự chọn, không dùng cap
    # max_rules, để giữ matched-budget với các heuristic.
    selected_path = os.path.join(output_dir, "selected_rules.pkl")
    with open(selected_path, "rb") as file:
        selected_rules = pickle.load(file)
    budget = min(len(selected_rules), n_rules)
    overlap_conf = topk_overlap(rank_p, rank_conf, budget)
    overlap_alone = topk_overlap(rank_p, rank_alone, budget)

    logger.info(
        "Spearman(p_include, confidence)=%.4f | Spearman(p_include, single_rule_reward)=%.4f",
        spearman_conf, spearman_alone,
    )
    logger.info(
        "Kendall-tau-b(p_include, confidence)=%.4f | Kendall-tau-b(p_include, single_rule_reward)=%.4f",
        kendall_conf, kendall_alone,
    )
    logger.info(
        "Top-%d overlap (Jaccard) p_include vs confidence=%.4f | vs single_rule_reward=%.4f",
        budget, overlap_conf, overlap_alone,
    )

    pos_p = {idx: pos for pos, idx in enumerate(rank_p)}
    pos_conf = {idx: pos for pos, idx in enumerate(rank_conf)}
    pos_alone = {idx: pos for pos, idx in enumerate(rank_alone)}

    rows = []
    for i, rule in enumerate(valid_rules):
        rows.append(
            {
                "rule_idx": i,
                "rule": str(rule),
                "confidence": rule.confidence,
                "p_include": p_include[i].item(),
                "single_rule_reward": singleton_scores[i].item(),
                "target_class": int(rule.target_class),
                "rule_length": len(rule.conditions),
                "rank_p_include": pos_p[i],
                "rank_confidence": pos_conf[i],
                "rank_single_rule_reward": pos_alone[i],
            }
        )
    df = pd.DataFrame(rows).sort_values("rank_p_include")
    csv_path = os.path.join(output_dir, "rule_ranking_analysis.csv")
    df.to_csv(csv_path, index=False)

    metrics_row = {
        "seed": int(params["seed"]),
        "backbone": params["baseline_comparison"][
            "selected_architecture"
        ],
        "loss_type": rule_order["loss_type"],
        "trajectory_samples": K,
        "n_rules": n_rules,
        "top_k_budget": budget,
        "spearman_inclusion_confidence": spearman_conf,
        "spearman_inclusion_single_rule_reward": spearman_alone,
        "kendall_tau_b_inclusion_confidence": kendall_conf,
        "kendall_tau_b_inclusion_single_rule_reward": kendall_alone,
        "top_k_jaccard_inclusion_confidence": overlap_conf,
        "top_k_jaccard_inclusion_single_rule_reward": overlap_alone,
    }
    metrics_path = os.path.join(
        output_dir, "rule_ranking_analysis_metrics.csv"
    )
    with open(metrics_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(metrics_row)
        )
        writer.writeheader()
        writer.writerow(metrics_row)

    summary_path = os.path.join(output_dir, "rule_ranking_analysis_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(
            f"K={K}, n_rules={n_rules}, "
            f"budget(actual_gflownet_elite)={budget}\n"
        )
        f.write(f"Spearman(p_include, confidence)       = {spearman_conf:.4f}\n")
        f.write(f"Spearman(p_include, single_rule_reward)= {spearman_alone:.4f}\n")
        f.write(f"Kendall-tau-b(p_include, confidence)   = {kendall_conf:.4f}\n")
        f.write(f"Kendall-tau-b(p_include, single_rule_reward)= {kendall_alone:.4f}\n")
        f.write(f"Top-{budget} Jaccard overlap vs confidence      = {overlap_conf:.4f}\n")
        f.write(f"Top-{budget} Jaccard overlap vs single_rule_reward = {overlap_alone:.4f}\n")

    logger.info(
        "Đã ghi %s, %s và %s",
        csv_path,
        metrics_path,
        summary_path,
    )
    logger.info("Stage 4b (phân tích ranking) hoàn thành.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--K", type=int, default=None, help="Số trajectory sample từ policy đã train (200-500 khuyến nghị). Mặc định: params.yaml -> rule_ranking_analysis.K")
    args = parser.parse_args()
    main(args.config, K=args.K)
