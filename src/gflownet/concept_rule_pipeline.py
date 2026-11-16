"""Pipeline: GFlowNet xây luật trên image_concept -> tập luật con ít xung đột -> ``RuleSet``.

Luồng:
  1. ``ConceptRuleEnv`` sinh luật ứng viên từng bước (TB hoặc DB, như pipeline cũ).
  2. Mỗi trajectory kết thúc được ghi vào ``_CandidateArchive`` (khử trùng lặp theo
     tập literal + vế phải), giữ điểm tốt nhất từng thấy.
  3. ``select_rule_set`` chọn tối đa ``max_rules`` luật theo điểm giảm dần, bỏ luật
     không mang thêm mẫu đúng mới hoặc xung đột quá ngưỡng với tập đã chọn.
  4. ``to_rule`` đổi literal thành ``Condition`` trên concept probability, nên tập
     luật dùng được NGAY bởi ``VectorizedRulePenalty`` và ``ConceptRuleHead``
     (cùng quy ước với luật Stage 3: feature_index = chỉ số concept).

Điểm khác với ``RuleExtractionPipeline``: ở đó GFlowNet CHỌN từ luật cây quyết định
đã có sẵn; ở đây GFlowNet TỰ TẠO luật từ nhãn concept.
"""
import json
import os
import pickle
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet import DBGFlowNet, TBGFlowNet
from gfn.utils.modules import MLP

from src.gflownet.concept_rule_env import ConceptRuleEnv
from src.gflownet.concept_rule_reward import ConceptRuleReward, Literal, cover_mask
from src.rules.rule_types import Condition, Rule, RuleSet
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Ngưỡng nhị phân hoá concept: concept probability > 0.5 coi như "có mặt".
CONCEPT_THRESHOLD = 0.5


@dataclass
class ConceptRuleSearchConfig:
    max_literals: int = 4
    max_rules: int = 30
    min_cover: int = 5
    # Chọn tập luật
    max_conflict: float = 0.2          # tỉ lệ mẫu phủ bởi luật khác vế phải tối đa cho phép
    min_new_correct: int = 3           # luật phải thêm ít nhất ngần này mẫu đúng mới chưa được phủ
    # Reward
    w_prec: float = 1.0
    w_rec: float = 0.5
    w_len: float = 0.1
    w_min_cover: float = 1.0
    m_estimate: float = 2.0
    beta: float = 3.0
    # GFlowNet
    loss_type: str = "tb"              # "tb" | "db"
    hidden_dim: int = 256
    num_iterations: int = 1000
    batch_size: int = 256
    lr: float = 1e-3
    logZ_lr: float = 0.1
    logZ_warmup_steps: int = 50
    grad_clip_max_norm: float = 5.0
    archive_size: int = 5000
    final_samples: int = 4000
    log_every: int = 100
    seed: Optional[int] = None


@dataclass
class ConceptRuleCandidate:
    literals: Tuple[Literal, ...]
    target: int
    precision: float
    recall: float
    n_covered: int
    score: float

    @property
    def key(self) -> Tuple[Tuple[Literal, ...], int]:
        return (tuple(sorted(self.literals)), self.target)


@dataclass
class ConceptRuleSearchResult:
    rules: List[Rule]
    candidates: List[ConceptRuleCandidate]          # tập đã chọn, theo thứ tự chọn
    archive_size: int
    policy_state_dict: Dict[str, torch.Tensor] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Chuyển đổi và hiển thị luật
# ----------------------------------------------------------------------
def literals_to_rule(
    literals: Sequence[Literal], target: int, confidence: float
) -> Rule:
    """Literal (k, 1) -> ``concept_k > 0.5``; (k, 0) -> ``concept_k <= 0.5``."""
    conditions: List[Condition] = []
    for concept_idx, polarity in sorted(literals):
        if polarity == 1:
            conditions.append(Condition(concept_idx, ">", CONCEPT_THRESHOLD))
        else:
            conditions.append(Condition(concept_idx, "<=", CONCEPT_THRESHOLD))
    return Rule(conditions=conditions, target_class=int(target), confidence=float(confidence))


def rule_to_text(
    literals: Sequence[Literal],
    target: int,
    concept_names: Sequence[str],
    class_names: Optional[Sequence[str]] = None,
) -> str:
    """Dạng đọc được: ``Áo dài ∧ Chùa ⇒ khmer`` (literal vắng mặt có dấu ¬)."""
    parts = []
    for concept_idx, polarity in sorted(literals):
        name = concept_names[concept_idx] if concept_idx < len(concept_names) else f"concept_{concept_idx}"
        parts.append(name if polarity == 1 else f"¬{name}")
    cls = class_names[target] if class_names and target < len(class_names) else f"class_{target}"
    return " ∧ ".join(parts) + f" ⇒ {cls}"


# ----------------------------------------------------------------------
# Archive ứng viên
# ----------------------------------------------------------------------
class _CandidateArchive:
    """Giữ luật ứng viên khác nhau theo điểm cao nhất (khử trùng lặp theo literal + vế phải)."""

    def __init__(self, reward: ConceptRuleReward, max_size: int) -> None:
        self.reward = reward
        self.max_size = max_size
        self._items: Dict[Tuple, ConceptRuleCandidate] = {}

    def update(self, terminal_states: torch.Tensor) -> None:
        """Nhận tensor terminal (B, D). Chỉ giữ các trạng thái có điểm cải thiện."""
        if terminal_states.numel() == 0:
            return
        uniq = torch.unique(terminal_states.float(), dim=0)
        comps = self.reward.components(uniq)
        scores = self.reward.score(uniq)
        for i, (literals, target) in enumerate(self.reward.literals_batch(uniq)):
            if not literals:
                continue
            cand = ConceptRuleCandidate(
                literals=tuple(literals),
                target=target,
                precision=float(comps["precision"][i]),
                recall=float(comps["recall"][i]),
                n_covered=int(comps["n_covered"][i]),
                score=float(scores[i]),
            )
            prev = self._items.get(cand.key)
            if prev is None or cand.score > prev.score:
                self._items[cand.key] = cand
        if len(self._items) > self.max_size * 2:
            kept = sorted(self._items.values(), key=lambda c: c.score, reverse=True)[: self.max_size]
            self._items = {c.key: c for c in kept}

    def ranked(self) -> List[ConceptRuleCandidate]:
        return sorted(self._items.values(), key=lambda c: c.score, reverse=True)


# ----------------------------------------------------------------------
# Chọn tập luật ít xung đột
# ----------------------------------------------------------------------
def select_rule_set(
    candidates: Sequence[ConceptRuleCandidate],
    concept_matrix: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    max_rules: int,
    max_conflict: float,
    min_new_correct: int,
) -> List[ConceptRuleCandidate]:
    """Tham lam theo điểm. Nhận luật nếu:
      - thêm được >= min_new_correct mẫu ĐÚNG mà các luật đã chọn chưa phủ đúng; và
      - tỉ lệ mẫu nó phủ mà một luật ĐÃ CHỌN khác vế phải cũng phủ <= max_conflict.
    Đây là phiên bản tập luật của xung đột: đo theo MẪU (không theo cặp luật) để
    không bị pha loãng khi tập lớn lên.
    """
    # Mọi tensor trung gian nằm cùng device với concept_matrix (có thể là GPU).
    device = concept_matrix.device
    labels = labels.to(device)
    n = concept_matrix.shape[0]
    covered_correct = torch.zeros(n, dtype=torch.bool, device=device)
    cover_by_class = torch.zeros(n, num_classes, dtype=torch.long, device=device)
    chosen: List[ConceptRuleCandidate] = []

    for cand in candidates:
        if len(chosen) >= max_rules:
            break
        mask = cover_mask(concept_matrix, cand.literals)
        n_cov = int(mask.sum())
        if n_cov == 0:
            continue
        correct = mask & (labels == cand.target)
        new_correct = int((correct & ~covered_correct).sum())
        if new_correct < min_new_correct:
            continue
        others_cover = (cover_by_class.sum(dim=1) - cover_by_class[:, cand.target]) > 0
        conflict = float((mask & others_cover).sum()) / n_cov
        if conflict > max_conflict:
            continue
        chosen.append(cand)
        covered_correct |= correct
        cover_by_class[:, cand.target] += mask.long()
    return chosen


# ----------------------------------------------------------------------
# Đánh giá tập luật trên một split (để báo cáo, không dùng để huấn luyện)
# ----------------------------------------------------------------------
def evaluate_rule_set(
    candidates: Sequence[ConceptRuleCandidate],
    concept_matrix: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """Voting theo confidence (= precision trên tập train): luật nào phủ mẫu thì bỏ phiếu
    cho vế phải của nó. Mẫu không luật nào phủ -> không dự đoán (đếm riêng)."""
    device = concept_matrix.device
    labels = labels.to(device)
    n = concept_matrix.shape[0]
    votes = torch.zeros(n, num_classes, device=device)
    for cand in candidates:
        mask = cover_mask(concept_matrix, cand.literals)
        votes[mask, cand.target] += cand.precision
    covered = votes.sum(dim=1) > 0
    preds = votes.argmax(dim=1)
    n_cov = int(covered.sum())
    correct = int(((preds == labels) & covered).sum())
    return {
        "n_samples": float(n),
        "coverage": n_cov / max(n, 1),
        "accuracy_on_covered": correct / max(n_cov, 1),
        "accuracy_overall_no_fallback": correct / max(n, 1),
    }


# ----------------------------------------------------------------------
# Pipeline GFlowNet
# ----------------------------------------------------------------------
class ConceptRuleGFlowNetPipeline:
    def __init__(self, device: str = "cpu", cfg: Optional[ConceptRuleSearchConfig] = None) -> None:
        self.device = torch.device(device)
        self.cfg = cfg or ConceptRuleSearchConfig()

    def _build_gflownet(self, env: ConceptRuleEnv):
        cfg = self.cfg
        pf_module = MLP(input_dim=env.dim, output_dim=env.n_actions, hidden_dim=cfg.hidden_dim,
                        n_hidden_layers=2, add_layer_norm=True)
        pb_module = MLP(input_dim=env.dim, output_dim=env.n_actions - 1, hidden_dim=cfg.hidden_dim,
                        n_hidden_layers=2, add_layer_norm=True)
        pf = DiscretePolicyEstimator(module=pf_module, n_actions=env.n_actions, is_backward=False,
                                     preprocessor=env.preprocessor)
        pb = DiscretePolicyEstimator(module=pb_module, n_actions=env.n_actions, is_backward=True,
                                     preprocessor=env.preprocessor)
        if cfg.loss_type == "tb":
            gflownet = TBGFlowNet(pf=pf, pb=pb, init_logZ=0.0)
            optimizer = torch.optim.Adam([
                {"params": list(gflownet.pf_pb_parameters()), "lr": cfg.lr, "weight_decay": 1e-5},
                {"params": list(gflownet.logz_parameters()), "lr": cfg.logZ_lr, "weight_decay": 0.0},
            ])
        elif cfg.loss_type == "db":
            logF_module = MLP(input_dim=env.dim, output_dim=1, hidden_dim=cfg.hidden_dim, n_hidden_layers=2)
            logF = ScalarEstimator(module=logF_module, preprocessor=env.preprocessor)
            gflownet = DBGFlowNet(pf=pf, pb=pb, logF=logF)
            optimizer = torch.optim.Adam([
                {"params": list(gflownet.pf_pb_parameters()), "lr": cfg.lr, "weight_decay": 1e-5},
                {"params": list(logF.parameters()), "lr": cfg.lr * 2, "weight_decay": 0.0},
            ])
        else:
            raise ValueError(f"loss_type phải là 'tb' hoặc 'db', nhận '{cfg.loss_type}'.")
        gflownet.to(self.device)
        return gflownet, optimizer

    def _train_step(self, gflownet, optimizer, env, in_warmup: bool) -> Tuple[torch.Tensor, object]:
        # Giống pipeline cũ: trong warmup chỉ học logZ (TB), giữ P_F/P_B đóng băng.
        for p in gflownet.pf_pb_parameters():
            p.requires_grad_(not in_warmup)
        trajectories = gflownet.sample_trajectories(env, n=self.cfg.batch_size, save_logprobs=True)
        samples = gflownet.to_training_samples(trajectories)
        optimizer.zero_grad()
        loss = gflownet.loss(env, samples, recalculate_all_logprobs=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gflownet.parameters(), max_norm=self.cfg.grad_clip_max_norm)
        optimizer.step()
        return loss, trajectories

    def run(
        self,
        concept_matrix: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        concept_names: Sequence[str],
        class_names: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
    ) -> ConceptRuleSearchResult:
        cfg = self.cfg
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        if concept_matrix.shape[1] != len(concept_names):
            raise ValueError(
                f"concept_matrix có {concept_matrix.shape[1]} cột nhưng có {len(concept_names)} tên concept."
            )

        A = concept_matrix.float().to(self.device)
        y = labels.long().to(self.device)
        reward = ConceptRuleReward(
            A, y, num_classes,
            max_literals=cfg.max_literals, w_prec=cfg.w_prec, w_rec=cfg.w_rec,
            w_len=cfg.w_len, w_min_cover=cfg.w_min_cover, min_cover=cfg.min_cover,
            m_estimate=cfg.m_estimate, beta=cfg.beta,
        )
        env = ConceptRuleEnv(A.shape[1], num_classes, cfg.max_literals, reward, device=self.device)
        gflownet, optimizer = self._build_gflownet(env)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.7, patience=20, min_lr=1e-6
        )
        archive = _CandidateArchive(reward, cfg.archive_size)

        logger.info(
            "ConceptRule GFlowNet: %d concepts, %d classes, %d mẫu, max_literals=%d, loss=%s",
            A.shape[1], num_classes, A.shape[0], cfg.max_literals, cfg.loss_type,
        )
        for it in range(cfg.num_iterations):
            in_warmup = it < cfg.logZ_warmup_steps
            loss, trajectories = self._train_step(gflownet, optimizer, env, in_warmup)
            archive.update(trajectories.terminating_states.tensor.detach())
            if (it + 1) % cfg.log_every == 0:
                avg_log_r = float(trajectories.log_rewards.mean())
                scheduler.step(avg_log_r)
                top = archive.ranked()[:1]
                best = f"{top[0].score:.3f}" if top else "n/a"
                logger.info(
                    "iter %d | loss=%.4f | avg_log_r=%.3f | archive=%d | best_score=%s",
                    it + 1, loss.item(), avg_log_r, len(archive._items), best,
                )

        # Lấy thêm mẫu cuối cùng từ policy đã học để không bỏ sót luật tốt.
        with torch.no_grad():
            n_batches = max(1, cfg.final_samples // max(cfg.batch_size, 1))
            for _ in range(n_batches):
                final = gflownet.sample_trajectories(env, n=cfg.batch_size, save_logprobs=False)
                archive.update(final.terminating_states.tensor.detach())

        ranked = archive.ranked()
        chosen = select_rule_set(
            ranked, A, y, num_classes,
            max_rules=cfg.max_rules, max_conflict=cfg.max_conflict,
            min_new_correct=cfg.min_new_correct,
        )
        rules = [literals_to_rule(c.literals, c.target, c.precision) for c in chosen]
        logger.info(
            "Chọn %d/%d luật từ archive %d ứng viên.", len(rules), cfg.max_rules, len(ranked)
        )

        result = ConceptRuleSearchResult(
            rules=rules,
            candidates=chosen,
            archive_size=len(ranked),
            policy_state_dict={k: v.detach().cpu().clone() for k, v in gflownet.state_dict().items()},
        )
        if output_dir is not None:
            self._save(result, output_dir, concept_names, class_names)
        return result

    def _save(
        self,
        result: ConceptRuleSearchResult,
        output_dir: str,
        concept_names: Sequence[str],
        class_names: Optional[Sequence[str]],
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "selected_rules.pkl"), "wb") as f:
            pickle.dump(RuleSet(rules=result.rules), f)
        torch.save(
            {
                "policy": result.policy_state_dict,
                "config": asdict(self.cfg),
                "n_concepts": len(concept_names),
            },
            os.path.join(output_dir, "gflownet_concept_policy.pth"),
        )
        readable = [
            {
                "text": rule_to_text(c.literals, c.target, concept_names, class_names),
                "literals": [
                    {"concept": int(k), "name": concept_names[k], "present": bool(p)}
                    for k, p in sorted(c.literals)
                ],
                "target_class": int(c.target),
                "precision": c.precision,
                "recall": c.recall,
                "n_covered": c.n_covered,
                "score": c.score,
            }
            for c in result.candidates
        ]
        with open(os.path.join(output_dir, "concept_rules.json"), "w", encoding="utf-8") as f:
            json.dump(readable, f, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, "concept_rules.txt"), "w", encoding="utf-8") as f:
            for item in readable:
                f.write(f"{item['text']}   [precision={item['precision']:.3f}, "
                        f"recall={item['recall']:.3f}, n={item['n_covered']}]\n")
