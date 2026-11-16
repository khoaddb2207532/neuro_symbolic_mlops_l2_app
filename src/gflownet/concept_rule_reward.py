"""Reward cho luật do GFlowNet XÂY DỰNG trên nhãn image_concept.

Một luật có dạng::

    (l_1 ∧ l_2 ∧ ... ∧ l_m)  =>  class c

trong đó mỗi literal là "concept k có mặt" (giá trị 1) hoặc "concept k vắng mặt"
(giá trị 0) theo nhãn ``image_concepts``. Nhãn NaN (ignore) không khớp literal nào,
nên một ảnh chỉ được coi là phủ bởi luật khi TẤT CẢ literal đều xác định và đúng.

    score = w_prec * precision_c
          + w_rec  * recall_c
          - w_len  * (m / max_literals)
          - w_min_cover * [số mẫu phủ < min_cover]

    precision_c: m-estimate  (TP + m·p0) / (n_phủ + m),  p0 = tần suất lớp c.
                 Shrink về tần suất lớp để luật phủ rất ít mẫu không đạt điểm cao giả tạo.
    recall_c:    TP / n_c — tỉ lệ mẫu thuộc lớp c mà luật phủ đúng.

    log_reward = beta · score   (GFlowNet học theo log-reward, giống RuleSetReward)

Reward này chỉ đo chất lượng MỘT luật. Việc ghép nhiều luật thành tập có xung đột
thấp nằm ở ``concept_rule_pipeline.select_rule_set``.
"""
from typing import Dict, List, Sequence, Tuple

import torch

Literal = Tuple[int, int]  # (concept_index, polarity) — polarity 1 = có mặt, 0 = vắng mặt


def cover_mask(concept_matrix: torch.Tensor, literals: Sequence[Literal]) -> torch.Tensor:
    """Mask bool (N,) — mẫu nào thoả MỌI literal. Nhãn NaN không thoả literal nào.

    Đây là định nghĩa "phủ" duy nhất trong module; reward, selection và đánh giá
    đều dùng hàm này để không lệch công thức.
    """
    n = concept_matrix.shape[0]
    mask = torch.ones(n, dtype=torch.bool, device=concept_matrix.device)
    for concept_idx, polarity in literals:
        mask &= concept_matrix[:, concept_idx] == float(polarity)
    if not literals:
        mask[:] = False
    return mask


class ConceptRuleReward:
    """Chấm điểm trạng thái terminal của ``ConceptRuleEnv``.

    Layout trạng thái (batch, D) với D = 2K + C — xem ``concept_rule_env.py``:
      - [0, 2K): literal, slot 2k = concept k có mặt, slot 2k+1 = concept k vắng mặt
      - [2K, 2K+C): head one-hot, class c của vế phải
    """

    def __init__(
        self,
        concept_matrix: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        max_literals: int = 4,
        w_prec: float = 1.0,
        w_rec: float = 0.5,
        w_len: float = 0.1,
        w_min_cover: float = 1.0,
        min_cover: int = 5,
        m_estimate: float = 2.0,
        beta: float = 3.0,
    ) -> None:
        if concept_matrix.ndim != 2:
            raise ValueError("concept_matrix phải có dạng [N, K].")
        if labels.shape[0] != concept_matrix.shape[0]:
            raise ValueError("labels và concept_matrix phải cùng số mẫu N.")
        if max_literals < 1:
            raise ValueError("max_literals phải >= 1.")
        if min_cover < 1:
            raise ValueError("min_cover phải >= 1.")

        A = concept_matrix.float()
        self.concept_matrix = A
        self.n_samples, self.n_concepts = A.shape
        self.num_classes = num_classes
        self.labels = labels.long()
        self.max_literals = max_literals
        self.w_prec = w_prec
        self.w_rec = w_rec
        self.w_len = w_len
        self.w_min_cover = w_min_cover
        self.min_cover = min_cover
        self.m_estimate = m_estimate
        self.beta = beta

        # NaN == 1 và NaN == 0 đều là False, nên ảnh không có nhãn không khớp literal nào.
        self.pos_match = (A == 1.0).float()  # (N, K)
        self.neg_match = (A == 0.0).float()  # (N, K)

        self.class_counts = torch.bincount(self.labels, minlength=num_classes).float()  # (C,)
        self.base_rate = self.class_counts / max(self.n_samples, 1)

    # ------------------------------------------------------------------
    # Giải mã trạng thái
    # ------------------------------------------------------------------
    def decode(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trả về (pos (B,K), neg (B,K), target (B,)) từ tensor trạng thái (B, D)."""
        K, C = self.n_concepts, self.num_classes
        flat = state.reshape(-1, 2 * K + C).float()
        lit = flat[:, : 2 * K].reshape(-1, K, 2)
        head = flat[:, 2 * K:]
        return lit[..., 0], lit[..., 1], head.argmax(dim=-1)

    def literals_batch(self, states: torch.Tensor) -> List[Tuple[List[Literal], int]]:
        """Đổi batch trạng thái terminal (B, D) thành [(literals, target)] — một lần duyệt tensor."""
        pos, neg, target = self.decode(states)
        pos_np = (pos > 0).cpu().numpy()
        neg_np = (neg > 0).cpu().numpy()
        tgt_np = target.cpu().numpy()
        out: List[Tuple[List[Literal], int]] = []
        for b in range(pos_np.shape[0]):
            literals: List[Literal] = []
            for k in range(self.n_concepts):
                if pos_np[b, k]:
                    literals.append((k, 1))
                if neg_np[b, k]:
                    literals.append((k, 0))
            out.append((literals, int(tgt_np[b])))
        return out

    # ------------------------------------------------------------------
    # Đánh giá
    # ------------------------------------------------------------------
    def components(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Thống kê từng luật trong batch. Tất cả tensor có shape (B,)."""
        pos, neg, target = self.decode(state)
        n_lit = pos.sum(-1) + neg.sum(-1)  # (B,)

        # sat[b, j] = số literal của luật b đúng trên mẫu j. Phủ khi sat == n_lit.
        sat = pos @ self.pos_match.T + neg @ self.neg_match.T  # (B, N)
        covered = (sat == n_lit.unsqueeze(-1)) & (n_lit.unsqueeze(-1) > 0)

        labels = self.labels.unsqueeze(0)  # (1, N)
        correct = covered & (labels == target.unsqueeze(-1))

        n_covered = covered.float().sum(-1)
        tp = correct.float().sum(-1)
        p0 = self.base_rate[target]
        precision = (tp + self.m_estimate * p0) / (n_covered + self.m_estimate)
        n_class = self.class_counts[target].clamp(min=1.0)
        recall = tp / n_class
        support = n_covered / max(self.n_samples, 1)

        return {
            "n_literals": n_lit,
            "target": target.float(),
            "n_covered": n_covered,
            "true_positive": tp,
            "precision": precision,
            "recall": recall,
            "support": support,
        }

    def score(self, state: torch.Tensor) -> torch.Tensor:
        c = self.components(state)
        under_cover = (c["n_covered"] < self.min_cover).float()
        return (
            self.w_prec * c["precision"]
            + self.w_rec * c["recall"]
            - self.w_len * c["n_literals"] / float(self.max_literals)
            - self.w_min_cover * under_cover
        )

    def log_reward(self, state: torch.Tensor) -> torch.Tensor:
        return self.beta * self.score(state)

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_reward(state))
