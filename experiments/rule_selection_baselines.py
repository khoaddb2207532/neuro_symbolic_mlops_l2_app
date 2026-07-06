"""GOI B - So sanh GFlowNet voi cac heuristic chon luat khac.

Muc tieu: chung minh GFlowNet chon tap luat tot hon 3 heuristic don gian
(random / top-k confidence / greedy coverage), tren CUNG mot tap luat hop le
dau vao (khong train lai CNN/RF, khong sua GFlowNet hien co).

============================================================================
CAP NHAT (so sanh CONG BANG voi GFlowNet) - 2 thay doi chinh so voi ban dau
============================================================================
1) THUOC DO CHINH DE SO SANH gio la `reward_score` (= gia tri tra ve boi
   chinh `RuleSetReward.score()` trong `src/gflownet/reward.py`, dung
   CUNG cong thuc + CUNG trong so w_acc/w_cov/w_red/w_comp/beta ma GFlowNet
   duoc huan luyen de toi da hoa) - KHONG con dung `f1_like` de ket luan
   "phuong phap nao tot hon" nua, vi `f1_like` (trong `evaluate_run`) la
   mot cong thuc KHAC (thien ve coverage) khong phai thu GFlowNet duoc
   thuong khi train. `f1_like`/`accuracy`/`coverage`/`redundancy` van duoc
   giu lai trong CSV chi de tham khao/giai thich, khong dung de xep hang.
   `reward_score` cao hon = phuong phap do "gioi" hon THEO DUNG tieu chi
   GFlowNet toi uu; `reward_value = exp(beta * reward_score)` la gia tri
   reward thuc te (thang GFlowNet nhin thay khi train), don dieu tang theo
   reward_score nen xep hang y het, chi de doc truc quan hon dai luong
   RuleSelectionEnv.log_reward truoc khi exp.

2) NGAN SACH SO LUAT (budget) cong bang: GFlowNet tu quyet dinh so luat no
   chon (<= max_rules trong params.yaml), thuong nho hon max_rules rat
   nhieu (vi so hang `- w_comp * complexity` trong reward.py phat cang
   NHIEU luat cang bi tru diem). Neu ep 3 heuristic dung DUNG max_rules
   (vd 48) trong khi GFlowNet chi dung it hon (vd 19), 3 heuristic bi thiet
   o so hang complexity ma khong lien quan gi den chat luong luat chon.
   Vi vay script nay chay MOI heuristic 2 LAN:
     - budget="full"    : k = max_rules (params.yaml -> gflownet.max_rules)
     - budget="matched" : k = so luat GFlowNet THUC SU chon o lan chay nay
   => so sanh cong bang nhat la doi chieu hang "gflownet" voi cac hang
      heuristic budget="matched" (cung so luat, cung thuoc do reward_score).

GIA DINH (assumptions) do khong co artefact nhi phan that de kiem tra:
  - `outputs/03_rules/raw_rules.pkl` chua mot doi tuong `RuleSet`
    (dung `pickle.load`) - giong cach `pipelines/stage4_select_rules_gflownet.py`
    truyen truc tiep ket qua load duoc vao
    `RuleValidator.validate_and_build_tensors(rule_set, ...)`.
  - `outputs/02_features/val_features.pt` va `val_labels.pt` la tensor luu
    bang `torch.load` (features: (N, n_features) float, labels: (N,) long),
    dung dinh dang duoc `pipelines/stage4_select_rules_gflownet.py` doc.
  - `Rule.confidence` (trong `src/rules/rule_types.py`) da duoc gan gia tri
    that (0..1) sau khi qua `RuleValidator.validate_and_build_tensors`.
  - Metric tham khao (`f1_like`, `accuracy`, `coverage`, `redundancy`) dung
    `src.gflownet.evaluation.evaluate_run`; metric CHINH `reward_score`/
    `reward_value` dung truc tiep `RuleSetReward.score()` cua CUNG 1
    `shared_reward_module` (khoi tao 1 lan voi cover/correct/rule_len/
    max_rules/trong so cua toan bo `valid_rules` goc) de dam bao index
    dung nhat quan giua 4 phuong phap - khong dung reward_module noi bo
    cua GFlowNet (no shuffle ban sao valid_rules/tensor truoc khi train
    nen index lech voi `valid_rules` goc dung cho 3 heuristic). Voi
    GFlowNet, danh sach `Rule` no tra ve duoc anh xa nguoc lai ve index
    trong `valid_rules` goc bang `id(rule)` (object identity - cac object
    Rule KHONG bi copy khi pipeline permute, chi permute thu tu list/tensor).

Chay:
    python -m experiments.rule_selection_baselines --config params.yaml

Ky vong output:
    outputs/06_rule_selection_baselines/rule_selection_baselines.csv
    - moi hang la 1 (phuong phap, budget). random_selection la trung binh
      5 lan chay (seed 0..4) kem std, cho MOI budget.
    Cot chinh: method, budget, so_luat, reward_score, reward_value.
    Cot tham khao: f1_like, accuracy, coverage, redundancy,
      thoi_gian_chay_giay (+ *_std cho rieng random_selection).

Cach doc ket qua:
    - So sanh CONG BANG: loc cac hang budget="matched" + hang "gflownet",
      xep hang theo `reward_score` giam dan - hang cao nhat la phuong phap
      "toi uu" gan nhat theo dung tieu chi GFlowNet duoc huan luyen.
    - Hang budget="full" giu lai de tham khao/debug (vd de thay ro
      f1_like thien vi coverage nhu the nao neu ep du 48 luat).
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

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
#    `max_rules` o day la NGAN SACH duoc truyen vao cho LAN GOI NAY (co the
#    la budget "full" hoac budget "matched" - xem main()), KHONG phai luon
#    luon la params.yaml -> gflownet.max_rules.
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
        gains = (cover_bool & ~covered.unsqueeze(0)).sum(dim=1).float()
        gains[selected_mask] = -1.0
        best_idx = int(torch.argmax(gains).item())
        if gains[best_idx].item() <= 0:
            break
        selected.append(best_idx)
        selected_mask[best_idx] = True
        covered |= cover_bool[best_idx]

    return selected


# ----------------------------------------------------------------------
# 2) Cham diem: dung DUNG cong thuc RuleSetReward.score() de so sanh cong
#    bang voi GFlowNet, thay vi chi dung f1_like cua evaluate_run.
# ----------------------------------------------------------------------

def _score_indices(
    indices: List[int],
    n_rules: int,
    reward_module: RuleSetReward,
    device,
) -> Tuple[float, float]:
    """Tra ve (raw_score, reward_value) cho 1 tap chon (indices vao
    valid_rules), dung DUNG cong thuc/trong so ma GFlowNet duoc huan luyen
    de toi uu (`RuleSetReward.score()` trong src/gflownet/reward.py)."""
    s = torch.zeros(1, n_rules, device=device)
    if indices:
        s[0, indices] = 1.0
    with torch.no_grad():
        raw_score = reward_module.score(s).item()
    reward_value = float(np.exp(reward_module.beta * raw_score))
    return raw_score, reward_value


def _rule_indices_by_identity(selected_rules: List[Rule], valid_rules: List[Rule]) -> List[int]:
    """GFlowNet tra ve list[Rule] (khong phai index). Anh xa nguoc ve index
    trong `valid_rules` goc bang object identity (id()) - cac object Rule
    KHONG bi copy khi RuleExtractionPipeline.run() permute, chi hoan vi thu
    tu list/tensor, nen id() van khop."""
    id_to_idx = {id(r): i for i, r in enumerate(valid_rules)}
    missing = 0
    out = []
    for r in selected_rules:
        idx = id_to_idx.get(id(r))
        if idx is None:
            missing += 1
            continue
        out.append(idx)
    if missing:
        logger.warning(
            "%d/%d luat GFlowNet chon khong khop id() voi valid_rules goc - "
            "kiem tra lai gia dinh ve object identity.", missing, len(selected_rules)
        )
    return out


def _time_call(fn: Callable[[], List[int]]) -> Tuple[List[int], float]:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


def _build_row(
    method: str,
    budget_label: str,
    budget_value: int,
    idx: List[int],
    elapsed: float,
    valid_rules: List[Rule],
    reward_module: RuleSetReward,
    device,
) -> Dict[str, object]:
    n_rules = len(valid_rules)
    selected = [valid_rules[i] for i in idx]
    metric = evaluate_run(selected, valid_rules, reward_module)
    raw_score, reward_value = _score_indices(idx, n_rules, reward_module, device)
    return {
        "method": method,
        "budget": budget_label,
        "budget_value": budget_value,
        "so_luat": metric["n_rules"],
        "reward_score": raw_score,
        "reward_value": reward_value,
        "f1_like": metric["f1_like"],
        "accuracy": metric["accuracy"],
        "coverage": metric["coverage"],
        "redundancy": metric["redundancy"],
        "thoi_gian_chay_giay": elapsed,
        "n_runs": 1,
    }


def _aggregate_random_rows(rows: List[Dict[str, object]], budget_label: str, budget_value: int) -> Dict[str, object]:
    def _mean(key):
        return float(np.mean([r[key] for r in rows]))

    def _std(key):
        return float(np.std([r[key] for r in rows]))

    return {
        "method": "random_selection",
        "budget": budget_label,
        "budget_value": budget_value,
        "so_luat": _mean("so_luat"),
        "so_luat_std": _std("so_luat"),
        "reward_score": _mean("reward_score"),
        "reward_score_std": _std("reward_score"),
        "reward_value": _mean("reward_value"),
        "reward_value_std": _std("reward_value"),
        "f1_like": _mean("f1_like"),
        "f1_like_std": _std("f1_like"),
        "accuracy": _mean("accuracy"),
        "accuracy_std": _std("accuracy"),
        "coverage": _mean("coverage"),
        "coverage_std": _std("coverage"),
        "redundancy": _mean("redundancy"),
        "redundancy_std": _std("redundancy"),
        "thoi_gian_chay_giay": _mean("thoi_gian_chay_giay"),
        "thoi_gian_chay_giay_std": _std("thoi_gian_chay_giay"),
        "n_runs": len(rows),
    }


def run_heuristics_at_budget(
    budget_label: str,
    k: int,
    valid_rules: List[Rule],
    cover: torch.Tensor,
    correct: torch.Tensor,
    rule_len: torch.Tensor,
    reward_module: RuleSetReward,
    device,
    n_random_runs: int = 5,
) -> List[Dict[str, object]]:
    """Chay ca 3 heuristic voi ngan sach so luat `k` (budget_label = "full"
    hoac "matched"), tra ve list cac hang ket qua (random_selection da gop
    mean+std qua n_random_runs seed)."""
    rows: List[Dict[str, object]] = []

    random_rows = []
    for seed in range(n_random_runs):
        idx, elapsed = _time_call(
            lambda seed=seed: random_selection(valid_rules, cover, correct, rule_len, k, seed=seed)
        )
        row = _build_row("random_selection", budget_label, k, idx, elapsed, valid_rules, reward_module, device)
        random_rows.append(row)
        logger.info("[budget=%s] random_selection seed=%d: %s", budget_label, seed, row)
    rows.append(_aggregate_random_rows(random_rows, budget_label, k))

    idx, elapsed = _time_call(lambda: topk_confidence(valid_rules, cover, correct, rule_len, k))
    row = _build_row("topk_confidence", budget_label, k, idx, elapsed, valid_rules, reward_module, device)
    logger.info("[budget=%s] topk_confidence: %s", budget_label, row)
    rows.append(row)

    idx, elapsed = _time_call(lambda: greedy_coverage(valid_rules, cover, correct, rule_len, k))
    row = _build_row("greedy_coverage", budget_label, k, idx, elapsed, valid_rules, reward_module, device)
    logger.info("[budget=%s] greedy_coverage: %s", budget_label, row)
    rows.append(row)

    return rows


# ----------------------------------------------------------------------
# 3) main()
# ----------------------------------------------------------------------

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
    max_rules_full = gfn_cfg["max_rules"]

    # 1 reward_module DUY NHAT dung chung cho TAT CA phuong phap/budget, de
    # dam bao cham diem cong bang tren CUNG mot thuoc do/CUNG thu tu
    # valid_rules (xem gia dinh trong docstring dau file).
    shared_reward_module = RuleSetReward(
        cover=cover,
        correct=correct,
        rule_len=rule_len,
        max_rules=max_rules_full,
        w_acc=gfn_cfg.get("w_acc", 1.0),
        w_cov=gfn_cfg.get("w_cov", 0.5),
        w_red=gfn_cfg.get("w_red", 0.3),
        w_comp=gfn_cfg.get("w_comp", 0.2),
        beta=gfn_cfg.get("beta", 3.0),
    )

    rows: List[Dict[str, object]] = []

    # --- (a) chay GFlowNet TRUOC, de biet no thuc su chon bao nhieu luat
    #     -> dung lam ngan sach "matched" cho 3 heuristic ---
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
            max_rules=max_rules_full,
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
    gfn_idx = _rule_indices_by_identity(gflownet_selected, valid_rules)
    n_gfn = len(gfn_idx)

    gfn_row = _build_row("gflownet", "self", max_rules_full, gfn_idx, elapsed, valid_rules, shared_reward_module, device)
    logger.info("gflownet (tu chon %d/%d luat): %s", n_gfn, max_rules_full, gfn_row)
    rows.append(gfn_row)

    # --- (b) 3 heuristic o budget "full" (= max_rules trong params.yaml) ---
    rows.extend(run_heuristics_at_budget(
        "full", max_rules_full, valid_rules, cover, correct, rule_len, shared_reward_module, device,
    ))

    # --- (c) 3 heuristic o budget "matched" (= dung so luat GFlowNet vua
    #     chon) - day la SO SANH CONG BANG NHAT voi hang "gflownet" ---
    if n_gfn > 0:
        rows.extend(run_heuristics_at_budget(
            "matched", n_gfn, valid_rules, cover, correct, rule_len, shared_reward_module, device,
        ))
    else:
        logger.warning("GFlowNet khong chon luat nao (n_gfn=0) - bo qua budget 'matched'.")

    # --- xuat CSV ---
    csv_path = os.path.join(output_dir, "rule_selection_baselines.csv")
    fieldnames = [
        "method", "budget", "budget_value", "so_luat", "so_luat_std",
        "reward_score", "reward_score_std", "reward_value", "reward_value_std",
        "f1_like", "f1_like_std", "accuracy", "accuracy_std",
        "coverage", "coverage_std", "redundancy", "redundancy_std",
        "thoi_gian_chay_giay", "thoi_gian_chay_giay_std", "n_runs",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info("Da ghi ket qua so sanh vao: %s", csv_path)
    logger.info("So sanh CONG BANG: xem cac hang budget='matched' + hang 'gflownet', xep hang theo reward_score.")
    for row in sorted(
        [r for r in rows if r.get("budget") in ("matched", "self")],
        key=lambda r: r["reward_score"], reverse=True,
    ):
        logger.info("  [%s | budget=%s, k=%s] reward_score=%.4f reward_value=%.4f f1_like=%.4f",
                     row["method"], row["budget"], row.get("budget_value"),
                     row["reward_score"], row["reward_value"], row["f1_like"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)