"""Phân tích SAU KHI GFlowNet đã huấn luyện xong — KHÔNG train lại bất kỳ
tham số nào. Dùng lại đúng policy checkpoint được chỉ định làm sampler
thuần (chỉ forward, `torch.no_grad()`), rồi:

  1. Sample K (200-500) trajectory từ policy đã train.
  2. p_include(i) = tần suất luật i xuất hiện trong K tập được sample
     (ước lượng Monte Carlo của marginal inclusion probability theo policy).
  3. So sánh ranking theo p_include với 2 ranking "ngây thơ":
       - `rank_topk_confidence`   : sắp theo `rule.confidence` giảm dần
                                    (giống hệt heuristic topk_confidence).
       - `rank_marginal_gain_alone`: sắp theo reward của tập CHỈ GỒM MỘT
                                    luật đứng một mình (singleton set) —
                                    đây chính là "gain" ở BƯỚC ĐẦU TIÊN của
                                    greedy (vì covered ban đầu rỗng nên gain
                                    của greedy ở bước đầu = score singleton).

QUAN TRỌNG VỀ INDEX: checkpoint policy chỉ chứa state_dict, KHÔNG chứa
permutation mà `RuleExtractionPipeline.run()` đã áp dụng lên `valid_rules`
trước khi train (xem `src/gflownet/pipeline.py::run()`). Action index i của
policy đã train tương ứng với `valid_rules[i]` SAU permutation đó — vì vậy
module này BẮT BUỘC nạp lại `gflownet_rule_order.pkl` (được `pipeline.py`
lưu cùng lúc với checkpoint) thay vì gọi lại `RuleValidator` từ đầu, để đảm
bảo đúng ánh xạ action -> luật.
"""
import os
import pickle
from typing import Dict, List

import numpy as np
import torch
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet import DBGFlowNet, FMGFlowNet, TBGFlowNet
from gfn.utils.modules import MLP

from src.gflownet.env import RuleSelectionEnv
from src.gflownet.reward import RuleSetReward
from src.rules.rule_types import Rule
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_rule_order(output_dir: str) -> Dict:
    """Nạp `gflownet_rule_order.pkl` (valid_rules đã permute + cover/correct/
    rule_len + cấu hình kiến trúc) đã được `RuleExtractionPipeline.run()` lưu
    cùng lúc với checkpoint policy."""
    path = os.path.join(output_dir, "gflownet_rule_order.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy {path}. File này được RuleExtractionPipeline.run() "
            "tự lưu khi chạy stage4 (select_rules_gflownet) — hãy chạy lại "
            "stage4 với bản pipeline.py đã cập nhật trước khi chạy phân tích "
            "này. Không thể suy ngược đúng ánh xạ action->luật chỉ từ "
            "checkpoint policy vì checkpoint đó không lưu permutation."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def rebuild_gflownet(rule_order: Dict, ckpt_path: str, device: torch.device):
    """Dựng lại ĐÚNG kiến trúc gflownet (pf/pb + logZ hoặc logF, tuỳ loss_type)
    khớp với checkpoint đã lưu, rồi `load_state_dict` — KHÔNG khởi tạo lại
    ngẫu nhiên xong train, chỉ nạp trọng số cũ để sample."""
    n_valid = rule_order["n_valid"]
    max_rules = rule_order["max_rules"]
    loss_type = rule_order["loss_type"]
    hidden_dim = rule_order["gfnet_hidden_dim"]

    cover = rule_order["cover"].to(device)
    correct = rule_order["correct"].to(device)
    rule_len = rule_order["rule_len"].to(device)
    labels = rule_order["labels"].to(device)
    valid_rules: List[Rule] = rule_order["valid_rules"]
    targets = torch.tensor([r.target_class for r in valid_rules], device=device)
    confidences = torch.tensor(
        [r.confidence for r in valid_rules], device=device
    )

    reward_module = RuleSetReward(
        cover=cover, correct=correct, rule_len=rule_len, max_rules=max_rules,
        targets=targets, labels=labels, confidences=confidences,
        w_acc=rule_order.get("w_acc", 1.0), w_cov=rule_order.get("w_cov", 0.5),
        w_wrong=rule_order.get("w_wrong", 0.75),
        w_conflict=rule_order.get("w_conflict", 0.1),
        beta=rule_order.get("beta", 3.0),
    )

    env = RuleSelectionEnv(
        n_valid, max_rules, reward_module, device=device
    )

    add_layer_norm = rule_order.get("policy_add_layer_norm", True)
    pf_module = MLP(
        input_dim=env.state_shape[-1],
        output_dim=env.n_actions,
        hidden_dim=hidden_dim,
        n_hidden_layers=2,
        add_layer_norm=add_layer_norm,
    )
    pb_module = MLP(
        input_dim=env.state_shape[-1],
        output_dim=env.n_actions - 1,
        hidden_dim=hidden_dim,
        n_hidden_layers=2,
        add_layer_norm=add_layer_norm,
    )
    pf_estimator = DiscretePolicyEstimator(module=pf_module, n_actions=env.n_actions, is_backward=False, preprocessor=env.preprocessor)
    pb_estimator = DiscretePolicyEstimator(module=pb_module, n_actions=env.n_actions, is_backward=True, preprocessor=env.preprocessor)

    if loss_type == "tb":
        gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)
    elif loss_type == "db":
        logF_module = MLP(input_dim=env.state_shape[-1], output_dim=1, hidden_dim=hidden_dim, n_hidden_layers=2)
        logF_estimator = ScalarEstimator(module=logF_module, preprocessor=env.preprocessor)
        gflownet = DBGFlowNet(pf=pf_estimator, pb=pb_estimator, logF=logF_estimator)
    elif loss_type == "fm":
        gflownet = FMGFlowNet(estimator=pf_estimator)
    else:
        raise ValueError(f"loss_type phải là 'tb'/'db'/'fm', nhận '{loss_type}'")

    gflownet.to(device)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint GFlowNet: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt.get("n_rules") != n_valid or ckpt.get("max_rules") != max_rules:
        raise ValueError(
            "Checkpoint GFlowNet không khớp rule order: "
            f"checkpoint(n_rules={ckpt.get('n_rules')}, max_rules={ckpt.get('max_rules')}) "
            f"!= rule_order(n_rules={n_valid}, max_rules={max_rules})."
        )
    gflownet.load_state_dict(ckpt["model"])
    gflownet.eval()
    for p in gflownet.parameters():
        p.requires_grad_(False)

    return gflownet, env, valid_rules, reward_module


@torch.no_grad()
def sample_inclusion_probabilities(
    gflownet, env: RuleSelectionEnv, n_rules: int, K: int = 300, sample_batch: int = 256
) -> torch.Tensor:
    """Sample K trajectory TỪ POLICY ĐÃ TRAIN (chỉ forward, không cập nhật
    gradient) -> p_include(i) = tần suất luật i xuất hiện trong K tập được
    sample. Chia thành nhiều batch nhỏ (`sample_batch`) để không tràn VRAM
    khi K lớn — kết quả không đổi so với sample thẳng K trong 1 lần."""
    counts = torch.zeros(n_rules, device=env.device)
    done = 0
    while done < K:
        b = min(sample_batch, K - done)
        traj = gflownet.sample_trajectories(env, n=b, save_logprobs=False)
        term = traj.terminating_states.tensor.bool()
        counts += term.float().sum(dim=0)
        done += b
    return (counts / K).cpu()


def rank_topk_confidence(valid_rules: List[Rule]) -> List[int]:
    """Ranking 'ngây thơ' #1 — giống hệt tiêu chí xếp hạng của
    `experiments/rule_selection_baselines.py::topk_confidence`, nhưng trả về
    FULL ranking (list index giảm dần theo confidence) thay vì chỉ top-k."""
    n = len(valid_rules)
    return sorted(range(n), key=lambda i: valid_rules[i].confidence, reverse=True)


def score_single_rules(
    reward_module: RuleSetReward,
    n_rules: int,
    device,
) -> torch.Tensor:
    """Raw ``RuleSetReward.score`` của từng singleton ruleset."""
    eye = torch.eye(n_rules, device=device)
    with torch.no_grad():
        return reward_module.score(eye).detach().cpu()


def rank_marginal_gain_alone(reward_module: RuleSetReward, n_rules: int, device) -> List[int]:
    """Ranking 'ngây thơ' #2 — điểm `reward_module.score()` của tập CHỈ GỒM
    MỘT luật i đứng một mình (singleton set), sắp giảm dần. Đây chính là
    "marginal gain của bước đầu tiên" trong greedy: `covered` ban đầu rỗng
    nên gain của greedy ở bước 1 = score của singleton {i} luôn, với MỌI i —
    do đó ranking theo singleton score = ranking theo "ai sẽ được greedy chọn
    đầu tiên nếu đứng một mình". Vector hoá toàn bộ n_rules cùng lúc bằng ma
    trận đơn vị (mỗi hàng = 1 singleton set) thay vì lặp qua từng luật."""
    scores = score_single_rules(reward_module, n_rules, device)
    return torch.argsort(scores, descending=True).tolist()


def ranking_from_scores(scores: torch.Tensor) -> List[int]:
    """Chuyển 1 vector điểm số (vd p_include) thành ranking (list index giảm dần)."""
    return torch.argsort(scores, descending=True).tolist()


def spearman_rho(rank_a: List[int], rank_b: List[int], n: int) -> float:
    """Spearman rank correlation giữa 2 ranking (không phụ thuộc scipy)."""
    if n <= 1:
        return float("nan")
    pos_a = np.empty(n)
    pos_b = np.empty(n)
    for pos, idx in enumerate(rank_a):
        pos_a[idx] = pos
    for pos, idx in enumerate(rank_b):
        pos_b[idx] = pos
    d2 = np.sum((pos_a - pos_b) ** 2)
    return float(1 - (6 * d2) / (n * (n ** 2 - 1)))


def kendall_tau(rank_a: List[int], rank_b: List[int], n: int) -> float:
    """Kendall's tau (O(n^2), đủ nhanh cho vài trăm-nghìn luật; không phụ
    thuộc scipy). Với n rất lớn, gọi hàm này có thể chậm — xem cảnh báo ở
    script orchestrate."""
    if n <= 1:
        return float("nan")
    pos_a = np.empty(n, dtype=np.int64)
    pos_b = np.empty(n, dtype=np.int64)
    for pos, idx in enumerate(rank_a):
        pos_a[idx] = pos
    for pos, idx in enumerate(rank_b):
        pos_b[idx] = pos
    concordant = discordant = 0
    for i in range(n):
        sa = pos_a[i + 1:] - pos_a[i]
        sb = pos_b[i + 1:] - pos_b[i]
        prod = sa * sb
        concordant += int((prod > 0).sum())
        discordant += int((prod < 0).sum())
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else float("nan")


def topk_overlap(rank_a: List[int], rank_b: List[int], k: int) -> float:
    """Tỉ lệ Jaccard giữa top-k của 2 ranking — trực quan hơn correlation khi
    người đọc chỉ quan tâm 'có chọn cùng 1 nhóm luật hay không' ở một ngân
    sách (budget) cụ thể."""
    if k <= 0:
        return float("nan")
    a = set(rank_a[:k])
    b = set(rank_b[:k])
    union = a | b
    return len(a & b) / len(union) if union else float("nan")
