"""GOI B - So sanh GFlowNet voi cac heuristic chon luat khac.

Muc tieu: chung minh GFlowNet chon tap luat tot hon 3 heuristic don gian
(random / top-k confidence / greedy coverage), tren CUNG mot tap luat hop le
dau vao (khong train lai CNN/RF, khong sua GFlowNet hien co).

GIA DINH (assumptions) do khong co artefact nhi phan that de kiem tra:
  - `outputs/03_rules/raw_rules.pkl` chua mot doi tuong `RuleSet`
    (dung `pickle.load`) - giong cach `pipelines/stage4_select_rules_gflownet.py`
    truyen truc tiep ket qua load duoc vao
    `RuleValidator.validate_and_build_tensors(rule_set, ...)`.
  - `outputs/02_features/val_features.pt` va `val_labels.pt` la tensor luu
    bang `torch.load` (features: (N, n_features) float, labels: (N,) long),
    dung dinh dang duoc `pipelines/stage4_select_rules_gflownet.py` doc.
  - `Rule.confidence` (trong `src/rules/rule_types.py`) da duoc gan gia tri
    that (0..1) sau khi qua `RuleValidator.validate_and_build_tensors`
    (dung la nhu vay: ham nay gan `rule.confidence = confs[idx].item()`
    truoc khi tra ve `filtered_rules`).
  - Metric so sanh dung `src.gflownet.evaluation.evaluate_run`, can mot
    `reward_module` (co `.cover`, `.correct`, `.jaccard`, `.max_rules`).
    Script nay tu dung lai `src.gflownet.reward.RuleSetReward` (khoi tao
    voi CUNG cover/correct/rule_len/max_rules/trong so) de dam bao 4
    phuong phap duoc cham diem tren CUNG mot thuoc do - khong dung reward
    module rieng cua tung lan chay GFlowNet (vi RuleExtractionPipeline
    shuffle valid_rules/cover/correct/rule_len noi bo truoc khi train,
    nen index cua reward_module noi bo cua GFlowNet KHONG con khop voi
    thu tu `valid_rules` goc dung cho 3 heuristic).

Chay:
    python -m experiments.rule_selection_baselines --config params.yaml

Ky vong output:
    outputs/06_rule_selection_baselines/rule_selection_baselines.csv
    - moi hang la 1 phuong phap (random_selection la trung binh 5 lan chay,
      seed 0..4, kem std).
    Cot: method, f1_like, coverage, redundancy, so_luat,
         thoi_gian_chay_giay (+ *_std cho rieng random_selection, cac
         phuong phap khac de trong / NaN vi chi chay 1 lan).

Cach doc ket qua:
    f1_like cao hon (o cung so_luat ~ max_rules) = phuong phap chon luat
    tot hon (can bang accuracy tren phan duoc phu va coverage). redundancy
    thap hon = it luat trung lap hon. So sanh hang "gflownet" voi 3 hang
    heuristic con lai de xem GFlowNet co vuot troi khong.
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import time
from typing import Callable, Dict, List

import numpy as np
import torch

from src.gflownet.evaluation import evaluate_run
from src.gflownet.pipeline import RuleExtractionPipeline
from src.gflownet.reward import RuleSetReward
from src.rules.rule_types import Rule
from src.rules.validator import RuleValidator
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# 1) Cac ham heuristic chon luat - CUNG chu ky, tra ve list[int] (indices
#    vao valid_rules / cover / correct / rule_len, KHONG phai vao raw_rules).
# ----------------------------------------------------------------------

def random_selection(
    valid_rules: List[Rule],
    cover: torch.Tensor,
    correct: torch.Tensor,
    rule_len: torch.Tensor,
    max_rules: int,
    seed: int = 0,
) -> List[int]:
    """Chon ngau nhien `max_rules` luat (khong lap lai)."""
    n_rules = len(valid_rules)
    k = min(max_rules, n_rules)
    rng = random.Random(seed)
    return rng.sample(range(n_rules), k)


def topk_confidence(
    valid_rules: List[Rule],
    cover: torch.Tensor,
    correct: torch.Tensor,
    rule_len: torch.Tensor,
    max_rules: int,
) -> List[int]:
    """Sap xep theo `rule.confidence` giam dan, lay top `max_rules`."""
    n_rules = len(valid_rules)
    k = min(max_rules, n_rules)
    order = sorted(range(n_rules), key=lambda i: valid_rules[i].confidence, reverse=True)
    return order[:k]


def greedy_coverage(
    valid_rules: List[Rule],
    cover: torch.Tensor,
    correct: torch.Tensor,
    rule_len: torch.Tensor,
    max_rules: int,
) -> List[int]:
    """Greedy submodular don gian: lap `max_rules` lan, moi lan chon luat
    lam tang so mau MOI duoc phu (chua tung duoc phu boi cac luat da chon)
    nhieu nhat. Dung lai neu khong con luat nao tang duoc coverage.
    Vector hoa tren GPU/CPU bang tensor `cover` (n_rules, n_val) bool,
    khong dung vong lap python long qua tung luat trong moi vong (chi
    vong lap ngoai qua `max_rules` buoc, moi buoc 1 phep tinh tensor)."""
    cover_bool = cover.bool()
    n_rules, n_val = cover_bool.shape
    device = cover_bool.device
    k = min(max_rules, n_rules)

    covered = torch.zeros(n_val, dtype=torch.bool, device=device)
    selected_mask = torch.zeros(n_rules, dtype=torch.bool, device=device)
    selected: List[int] = []

    for _ in range(k):
        # so mau moi ma moi luat con lai se phu them neu duoc chon
        gains = (cover_bool & ~covered.unsqueeze(0)).sum(dim=1).float()
        gains[selected_mask] = -1.0  # loai luat da chon
        best_idx = int(torch.argmax(gains).item())
        if gains[best_idx].item() <= 0:
            break  # khong luat nao con tang coverage -> dung som
        selected.append(best_idx)
        selected_mask[best_idx] = True
        covered |= cover_bool[best_idx]

    return selected


# ----------------------------------------------------------------------
# 2) main(): doc du lieu tinh, chay 3 heuristic + goi lai pipeline GFlowNet
#    co san, cham diem tat ca bang CUNG 1 reward_module/evaluate_run.
# ----------------------------------------------------------------------

def _time_call(fn: Callable[[], List[int]]) -> tuple[List[int], float]:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


def main(params_path: str = "params.yaml") -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    features_dir = os.path.join(params["output_dir"], "02_features")
    rules_dir = os.path.join(params["output_dir"], "03_rules")
    output_dir = os.path.join(params["output_dir"], "06_rule_selection_baselines")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(rules_dir, "raw_rules.pkl"), "rb") as f:
        raw_rules = pickle.load(f)

    val_features = torch.load(os.path.join(features_dir, "val_features.pt")).to(device)
    val_labels = torch.load(os.path.join(features_dir, "val_labels.pt")).to(device)

    validator = RuleValidator(
        min_supp=params["rules"]["min_support"],
        min_conf=params["rules"]["min_confidence"],
    )
    valid_rule_set, cover, correct, rule_len = validator.validate_and_build_tensors(
        raw_rules, val_features, val_labels, store_device=device
    )
    valid_rules = list(valid_rule_set.rules)
    logger.info("So luat hop le sau khi loc bang val set: %d", len(valid_rules))

    if not valid_rules:
        logger.warning("Khong co luat hop le nao - dung script.")
        return

    gfn_cfg = params["gflownet"]
    max_rules = gfn_cfg["max_rules"]

    # 1 reward_module DUY NHAT dung chung cho ca 4 phuong phap, de dam bao
    # cham diem cong bang tren CUNG 1 thuoc do/CUNG 1 thu tu valid_rules.
    shared_reward_module = RuleSetReward(
        cover=cover,
        correct=correct,
        rule_len=rule_len,
        max_rules=max_rules,
        w_acc=gfn_cfg.get("w_acc", 1.0),
        w_cov=gfn_cfg.get("w_cov", 0.5),
        w_red=gfn_cfg.get("w_red", 0.3),
        w_comp=gfn_cfg.get("w_comp", 0.2),
        beta=gfn_cfg.get("beta", 3.0),
    )

    rows: List[Dict[str, object]] = []

    # --- random_selection: chay nhieu lan (seed 0..4) lay mean + std ---
    n_random_runs = 5
    random_metrics: Dict[str, List[float]] = {
        "f1_like": [], "coverage": [], "redundancy": [], "n_rules": [], "time_s": [],
    }
    for seed in range(n_random_runs):
        idx, elapsed = _time_call(
            lambda seed=seed: random_selection(valid_rules, cover, correct, rule_len, max_rules, seed=seed)
        )
        selected = [valid_rules[i] for i in idx]
        metric = evaluate_run(selected, valid_rules, shared_reward_module)
        random_metrics["f1_like"].append(metric["f1_like"])
        random_metrics["coverage"].append(metric["coverage"])
        random_metrics["redundancy"].append(metric["redundancy"])
        random_metrics["n_rules"].append(metric["n_rules"])
        random_metrics["time_s"].append(elapsed)
        logger.info("random_selection seed=%d: %s (%.3fs)", seed, metric, elapsed)

    rows.append({
        "method": "random_selection",
        "f1_like": float(np.mean(random_metrics["f1_like"])),
        "f1_like_std": float(np.std(random_metrics["f1_like"])),
        "coverage": float(np.mean(random_metrics["coverage"])),
        "coverage_std": float(np.std(random_metrics["coverage"])),
        "redundancy": float(np.mean(random_metrics["redundancy"])),
        "redundancy_std": float(np.std(random_metrics["redundancy"])),
        "so_luat": float(np.mean(random_metrics["n_rules"])),
        "so_luat_std": float(np.std(random_metrics["n_rules"])),
        "thoi_gian_chay_giay": float(np.mean(random_metrics["time_s"])),
        "thoi_gian_chay_giay_std": float(np.std(random_metrics["time_s"])),
        "n_runs": n_random_runs,
    })

    # --- topk_confidence: chay 1 lan (deterministic) ---
    idx, elapsed = _time_call(
        lambda: topk_confidence(valid_rules, cover, correct, rule_len, max_rules)
    )
    selected = [valid_rules[i] for i in idx]
    metric = evaluate_run(selected, valid_rules, shared_reward_module)
    logger.info("topk_confidence: %s (%.3fs)", metric, elapsed)
    rows.append({
        "method": "topk_confidence",
        "f1_like": metric["f1_like"],
        "f1_like_std": None,
        "coverage": metric["coverage"],
        "coverage_std": None,
        "redundancy": metric["redundancy"],
        "redundancy_std": None,
        "so_luat": metric["n_rules"],
        "so_luat_std": None,
        "thoi_gian_chay_giay": elapsed,
        "thoi_gian_chay_giay_std": None,
        "n_runs": 1,
    })

    # --- greedy_coverage: chay 1 lan (deterministic) ---
    idx, elapsed = _time_call(
        lambda: greedy_coverage(valid_rules, cover, correct, rule_len, max_rules)
    )
    selected = [valid_rules[i] for i in idx]
    metric = evaluate_run(selected, valid_rules, shared_reward_module)
    logger.info("greedy_coverage: %s (%.3fs)", metric, elapsed)
    rows.append({
        "method": "greedy_coverage",
        "f1_like": metric["f1_like"],
        "f1_like_std": None,
        "coverage": metric["coverage"],
        "coverage_std": None,
        "redundancy": metric["redundancy"],
        "redundancy_std": None,
        "so_luat": metric["n_rules"],
        "so_luat_std": None,
        "thoi_gian_chay_giay": elapsed,
        "thoi_gian_chay_giay_std": None,
        "n_runs": 1,
    })

    # --- gflownet: goi lai pipeline co san (khong sua gflownet) ---
    # Luu y: RuleExtractionPipeline.run() tu shuffle ban sao cua
    # valid_rules/cover/correct/rule_len ben trong no va tra ve DANH SACH
    # DOI TUONG Rule (khong phai index) da duoc chon - nen ta dung thang
    # danh sach Rule nay voi evaluate_run(..., valid_rules, ...) ma KHONG
    # can index lai (evaluate_run tu map bang id(rule) -> vi tri trong
    # `valid_rules` goc, xem src/gflownet/evaluation.py).
    gfn_output_dir = os.path.join(output_dir, "gflownet_run")
    os.makedirs(gfn_output_dir, exist_ok=True)
    gfn_pipeline = RuleExtractionPipeline(
        device=device,
        w_acc=gfn_cfg.get("w_acc", 1.0),
        w_cov=gfn_cfg.get("w_cov", 0.5),
        w_red=gfn_cfg.get("w_red", 0.3),
        w_comp=gfn_cfg.get("w_comp", 0.2),
        beta=gfn_cfg.get("beta", 3.0),
    )

    def _run_gflownet() -> List[Rule]:
        return gfn_pipeline.run(
            valid_rules=valid_rules,
            cover=cover,
            correct=correct,
            rule_len=rule_len,
            max_rules=max_rules,
            output_dir=gfn_output_dir,
            gfnet_hidden_dim=gfn_cfg["hidden_dim"],
            num_iterations=gfn_cfg["num_iterations"],
            batch_size=gfn_cfg["batch_size"],
            lr=gfn_cfg["lr"],
            logZ_lr=gfn_cfg["logZ_lr"],
            device=device,
            validation_interval=gfn_cfg["validation_interval"],
            loss_type=gfn_cfg["loss_type"],
            logZ_warmup_steps=gfn_cfg["logZ_warmup_steps"],
            val_samples=gfn_cfg["val_samples"],
        )

    start = time.perf_counter()
    gflownet_selected = _run_gflownet()
    elapsed = time.perf_counter() - start
    metric = evaluate_run(gflownet_selected, valid_rules, shared_reward_module)
    logger.info("gflownet: %s (%.3fs)", metric, elapsed)
    rows.append({
        "method": "gflownet",
        "f1_like": metric["f1_like"],
        "f1_like_std": None,
        "coverage": metric["coverage"],
        "coverage_std": None,
        "redundancy": metric["redundancy"],
        "redundancy_std": None,
        "so_luat": metric["n_rules"],
        "so_luat_std": None,
        "thoi_gian_chay_giay": elapsed,
        "thoi_gian_chay_giay_std": None,
        "n_runs": 1,
    })

    # --- xuat CSV (khong dung pandas de tranh phu thuoc them - chi can
    # csv chuan cua python) ---
    import csv

    csv_path = os.path.join(output_dir, "rule_selection_baselines.csv")
    fieldnames = [
        "method", "f1_like", "f1_like_std", "coverage", "coverage_std",
        "redundancy", "redundancy_std", "so_luat", "so_luat_std",
        "thoi_gian_chay_giay", "thoi_gian_chay_giay_std", "n_runs",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info("Da ghi ket qua so sanh vao: %s", csv_path)
    for row in rows:
        logger.info("  %s", row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)