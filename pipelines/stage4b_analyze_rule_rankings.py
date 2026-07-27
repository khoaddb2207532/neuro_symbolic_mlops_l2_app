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
import os

import pandas as pd
import torch

from src.gflownet.rule_ranking_analysis import (
    kendall_tau,
    load_rule_order,
    rank_marginal_gain_alone,
    rank_topk_confidence,
    ranking_from_scores,
    rebuild_gflownet,
    sample_inclusion_probabilities,
    spearman_rho,
    topk_overlap,
)
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

# Kendall-tau ở đây là O(n^2) (thuần Python/numpy, không phụ thuộc scipy) —
# với vài nghìn luật vẫn ổn, nhưng để tránh script treo nếu n_rules quá lớn,
# ta bỏ qua kendall-tau (giữ lại spearman, vẫn O(n log n)) khi vượt ngưỡng này.
_KENDALL_MAX_N = 4000


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
    rank_alone = rank_marginal_gain_alone(reward_module, n_rules, device)

    spearman_conf = spearman_rho(rank_p, rank_conf, n_rules)
    spearman_alone = spearman_rho(rank_p, rank_alone, n_rules)

    if n_rules <= _KENDALL_MAX_N:
        kendall_conf = kendall_tau(rank_p, rank_conf, n_rules)
        kendall_alone = kendall_tau(rank_p, rank_alone, n_rules)
    else:
        logger.warning(
            "n_rules=%d > %d — bỏ qua kendall-tau (O(n^2)) để tránh chạy quá lâu, chỉ báo cáo spearman.",
            n_rules, _KENDALL_MAX_N,
        )
        kendall_conf = kendall_alone = float("nan")

    # Overlap top-k tại budget = max_rules (ngân sách GFlowNet được phép chọn) —
    # trực quan hơn correlation: "cùng chọn bao nhiêu luật trong top-budget?"
    budget = min(rule_order["max_rules"], n_rules)
    overlap_conf = topk_overlap(rank_p, rank_conf, budget)
    overlap_alone = topk_overlap(rank_p, rank_alone, budget)

    logger.info(
        "Spearman(p_include, confidence)=%.4f | Spearman(p_include, marginal_alone)=%.4f",
        spearman_conf, spearman_alone,
    )
    logger.info(
        "Kendall-tau(p_include, confidence)=%.4f | Kendall-tau(p_include, marginal_alone)=%.4f",
        kendall_conf, kendall_alone,
    )
    logger.info(
        "Top-%d overlap (Jaccard) p_include vs confidence=%.4f | vs marginal_alone=%.4f",
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
                "rank_p_include": pos_p[i],
                "rank_confidence": pos_conf[i],
                "rank_marginal_alone": pos_alone[i],
            }
        )
    df = pd.DataFrame(rows).sort_values("rank_p_include")
    csv_path = os.path.join(output_dir, "rule_ranking_analysis.csv")
    df.to_csv(csv_path, index=False)

    summary_path = os.path.join(output_dir, "rule_ranking_analysis_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"K={K}, n_rules={n_rules}, budget(max_rules)={budget}\n")
        f.write(f"Spearman(p_include, confidence)       = {spearman_conf:.4f}\n")
        f.write(f"Spearman(p_include, marginal_alone)   = {spearman_alone:.4f}\n")
        f.write(f"Kendall-tau(p_include, confidence)    = {kendall_conf:.4f}\n")
        f.write(f"Kendall-tau(p_include, marginal_alone)= {kendall_alone:.4f}\n")
        f.write(f"Top-{budget} Jaccard overlap vs confidence      = {overlap_conf:.4f}\n")
        f.write(f"Top-{budget} Jaccard overlap vs marginal_alone  = {overlap_alone:.4f}\n")

    logger.info("Đã ghi %s và %s", csv_path, summary_path)
    logger.info("Stage 4b (phân tích ranking) hoàn thành.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--K", type=int, default=None, help="Số trajectory sample từ policy đã train (200-500 khuyến nghị). Mặc định: params.yaml -> rule_ranking_analysis.K")
    args = parser.parse_args()
    main(args.config, K=args.K)
