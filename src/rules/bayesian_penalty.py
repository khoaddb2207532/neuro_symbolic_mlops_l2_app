"""Bayesian marginalization cho rule-penalty regularization.

Ý TƯỞNG: `VectorizedRulePenalty` (src/rules/penalty.py) phạt CNN theo MỘT
tập luật CỐ ĐỊNH (`selected_rules` — xấp xỉ MAP/best-found của GFlowNet).
Module này thay bằng kỳ vọng theo TOÀN BỘ PHÂN PHỐI luật mà policy GFlowNet
đã học được (posterior over rule-sets):

    L_bayes = E_{s ~ pi_theta(.)} [ L_rule_penalty(s) ]

ước lượng bằng Monte Carlo KHÔNG CHỆCH (unbiased) với K mẫu MỚI mỗi bước:

    L_bayes ≈ (1/K) * sum_{k=1}^{K} L_rule_penalty(s_k),   s_k ~ pi_theta i.i.d,
                                                            resample MỖI bước.

FROZEN SAMPLER: `pi_theta` là policy GFlowNet đã train xong (nạp từ
checkpoint sampler, với state_dict được lưu trong pipeline GFlowNet ở
`src/gflownet/pipeline.py`, dòng ~84-94). Toàn bộ tham số của nó bị
`requires_grad_(False)` và luôn `.eval()` — GFlowNet KHÔNG được cập nhật
trong lúc train CNN, chỉ đóng vai trò sampler thuần.

TỐI ƯU TÍNH TOÁN (không đổi công thức, chỉ đổi CÁCH TÍNH cho nhanh): trong
`VectorizedRulePenalty.forward`, với MỘT ruleset cố định gồm R luật, phần
"avg_loss theo từng luật riêng lẻ" — gọi là `avg_loss_full[i]` — chỉ phụ
thuộc vào luật i, batch hiện tại (features/logits) và nhiệt độ hiện tại,
KHÔNG phụ thuộc những luật nào khác có mặt trong ruleset. Vì vậy, với K mask
nhị phân m_k (mỗi mask là 1 ruleset được sample), ta có đẳng thức:

    L_rule_penalty(s_k) = penalty_weight * mean_{i in s_k}( avg_loss_full[i] )

nên chỉ cần tính `avg_loss_full` MỘT LẦN trên TOÀN BỘ universe `valid_rules`
(luật đã qua RuleValidator, không phải chỉ các luật GFlowNet chọn), rồi
average có trọng số mask cho K ruleset CÙNG LÚC bằng một phép nhân ma trận —
kết quả TOÁN HỌC GIỐNG HỆT việc build lại K module `VectorizedRulePenalty`
riêng biệt (mỗi cái ứng với 1 ruleset) rồi lấy trung bình, chỉ nhanh hơn rất
nhiều vì phần tốn kém nhất (tính rule_sat từ features) chỉ làm 1 lần thay vì
K lần.
"""
import json
import os
import pickle
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn

from src.rules.rule_types import Rule
from src.rules.temperature import geometric_temperature


class BayesianRuleMarginalization(nn.Module):
    """Drop-in thay thế cho `VectorizedRulePenalty` trong `train_one_epoch`
    (cùng chữ ký `forward(features, logits) -> scalar`, cùng có
    `update_temperature()` và `last_coverage_stats()`), nhưng phạt theo kỳ
    vọng Monte Carlo qua K ruleset resample mỗi bước từ policy GFlowNet đã
    đóng băng, thay vì 1 ruleset cố định.
    """

    def __init__(
        self,
        valid_rules: List[Rule],
        gflownet,
        env,
        K: int = 32,
        penalty_weight: float = 0.1,
        use_confidence: bool = True,
        smoothing: float = 0.1,
        num_classes: int = 12,
        initial_temp: float = 2.0,
        final_temp: float = 15.0,
        temp_warmup_epochs: int = 2,
        temp_anneal_epochs: int = 10,
    ):
        super().__init__()
        if len(valid_rules) != env.n_rules:
            raise ValueError(
                f"valid_rules ({len(valid_rules)} luật) không khớp env.n_rules "
                f"({env.n_rules}) — phải dùng ĐÚNG valid_rules đã lưu trong "
                "gflownet_rule_order.pkl (SAU permutation lúc train GFlowNet), "
                "nếu không mask sample sẽ trỏ nhầm luật."
            )
        self.valid_rules = valid_rules
        self.gflownet = gflownet
        self.env = env
        self.K = K
        self.penalty_weight = penalty_weight
        self.use_confidence = use_confidence
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.temp_warmup_epochs = temp_warmup_epochs
        self.temp_anneal_epochs = temp_anneal_epochs

        # ---- Frozen sampler: KHÔNG BAO GIỜ cập nhật gradient của GFlowNet
        # trong lúc train CNN — chỉ dùng để sample. ----
        self.gflownet.eval()
        for p in self.gflownet.parameters():
            p.requires_grad_(False)

        R = len(valid_rules)
        max_conds = max((len(r.conditions) for r in valid_rules), default=1)
        feat_idx = torch.zeros(R, max_conds, dtype=torch.long)
        thresholds = torch.zeros(R, max_conds, dtype=torch.float)
        ops = torch.zeros(R, max_conds, dtype=torch.bool)
        valid_m = torch.zeros(R, max_conds, dtype=torch.bool)
        targets = torch.zeros(R, dtype=torch.long)
        confs = torch.zeros(R, dtype=torch.float)
        for i, rule in enumerate(valid_rules):
            targets[i] = rule.target_class
            confs[i] = rule.confidence
            for j, cond in enumerate(rule.conditions):
                feat_idx[i, j] = cond.feature_index
                thresholds[i, j] = cond.threshold
                ops[i, j] = cond.operator == ">"
                valid_m[i, j] = True

        self.register_buffer("feat_idx", feat_idx)
        self.register_buffer("thresholds", thresholds)
        self.register_buffer("ops", ops)
        self.register_buffer("valid_m", valid_m)
        self.register_buffer("targets", targets)
        self.register_buffer("confs", confs)
        self.register_buffer("_temperature", torch.tensor(float(initial_temp)))
        # Thống kê gộp trên GPU để không phải ghi file/đồng bộ CPU ở mỗi batch.
        # persistent=False vì đây là audit của một lần train CNN, không phải
        # trạng thái cần nạp lại cùng checkpoint mô hình.
        self.register_buffer(
            "_sample_counts", torch.zeros(R, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_sampled_ruleset_count", torch.zeros((), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_sampled_rule_count", torch.zeros((), dtype=torch.long), persistent=False
        )

        self._last_rule_sat: Optional[torch.Tensor] = None
        self._last_masks: Optional[torch.Tensor] = None

    def update_temperature(self, epoch: int) -> None:
        """Cùng lịch cố định với ``VectorizedRulePenalty``."""
        new_temp = geometric_temperature(
            epoch,
            self.initial_temp,
            self.final_temp,
            self.temp_warmup_epochs,
            self.temp_anneal_epochs,
        )
        self._temperature.fill_(new_temp)

    @torch.no_grad()
    def _sample_masks(self, device) -> torch.Tensor:
        """Resample K tập luật MỚI từ policy đã đóng băng — gọi lại MỖI lần
        `forward()` (tức MỖI BƯỚC huấn luyện CNN), không cache giữa các batch,
        đúng yêu cầu 'resample mỗi bước'. `torch.no_grad()` vì đây chỉ là
        sampler, không lan truyền gradient ngược vào GFlowNet."""
        traj = self.gflownet.sample_trajectories(self.env, n=self.K, save_logprobs=False)
        masks = traj.terminating_states.tensor.bool().to(device)  # (K, R)
        self._sample_counts.add_(masks.sum(dim=0, dtype=torch.long))
        self._sampled_ruleset_count.add_(masks.size(0))
        self._sampled_rule_count.add_(masks.sum(dtype=torch.long))
        return masks

    def save_sampling_summary(self, output_dir: str) -> dict:
        """Lưu các luật từng được GFlowNet sample và tần suất inclusion.

        Không ghi từng trajectory trong vòng train vì K ruleset được sample ở
        mọi batch có thể tạo log rất lớn. Thay vào đó lưu union các luật dưới
        dạng pickle và bảng thống kê đầy đủ theo đúng thứ tự rule của GFlowNet.
        """
        os.makedirs(output_dir, exist_ok=True)
        counts = self._sample_counts.detach().cpu().tolist()
        total_rulesets = int(self._sampled_ruleset_count.item())
        total_rule_selections = int(self._sampled_rule_count.item())
        sampled_indices = [idx for idx, count in enumerate(counts) if count > 0]
        sampled_rules = [self.valid_rules[idx] for idx in sampled_indices]

        pickle_path = os.path.join(output_dir, "gflownet_sampled_rules.pkl")
        with open(pickle_path, "wb") as file:
            pickle.dump(sampled_rules, file)

        rows = []
        for idx, (rule, count) in enumerate(zip(self.valid_rules, counts)):
            rows.append(
                {
                    "Rule Index": idx,
                    "Sample Count": count,
                    "Inclusion Probability": (
                        count / total_rulesets if total_rulesets else 0.0
                    ),
                    "Ever Sampled": count > 0,
                    "Target Class": rule.target_class,
                    "Confidence (%)": round(rule.confidence * 100, 2),
                    "#Features": len(rule.conditions),
                    "Rule": str(rule),
                }
            )
        excel_path = os.path.join(output_dir, "gflownet_sampled_rules.xlsx")
        table = pd.DataFrame(
            rows,
            columns=[
                "Rule Index",
                "Sample Count",
                "Inclusion Probability",
                "Ever Sampled",
                "Target Class",
                "Confidence (%)",
                "#Features",
                "Rule",
            ],
        )
        if not table.empty:
            table = table.sort_values(
                ["Sample Count", "Rule Index"], ascending=[False, True]
            )
        table.to_excel(excel_path, index=False)

        summary = {
            "total_sampled_rulesets": total_rulesets,
            "total_rule_selections": total_rule_selections,
            "mean_ruleset_size": (
                total_rule_selections / total_rulesets if total_rulesets else 0.0
            ),
            "n_valid_rules": len(self.valid_rules),
            "n_rules_ever_sampled": len(sampled_rules),
            "pickle_path": pickle_path,
            "excel_path": excel_path,
        }
        summary_path = os.path.join(output_dir, "gflownet_sampling_summary.json")
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        return summary

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        R = len(self.valid_rules)
        if R == 0:
            self._last_rule_sat = None
            self._last_masks = None
            return torch.tensor(0.0, device=features.device)

        device = features.device
        batch_size = features.size(0)

        # ---- avg_loss_full: tính 1 lần cho TOÀN BỘ universe valid_rules,
        # y hệt công thức trong VectorizedRulePenalty.forward, chỉ khác là
        # không giới hạn ở 1 ruleset cụ thể. ----
        sel = features[:, self.feat_idx]  # (batch, R, max_conds)
        temp = self._temperature
        sat_le = torch.sigmoid(temp * (self.thresholds - sel))
        sat_gt = torch.sigmoid(temp * (sel - self.thresholds))
        cond_sat = torch.where(self.ops, sat_gt, sat_le)
        cond_sat = torch.where(self.valid_m, cond_sat, torch.ones_like(cond_sat))
        rule_sat = cond_sat.prod(dim=-1)  # (batch, R)
        self._last_rule_sat = rule_sat.detach()

        log_probs = torch.log_softmax(logits, dim=1)
        tgt_log_probs = log_probs.gather(1, self.targets.unsqueeze(0).expand(batch_size, -1))
        sum_log_probs = log_probs.sum(dim=1, keepdim=True).expand(-1, R)
        log_loss = -(
            (1.0 - self.smoothing) * tgt_log_probs + (self.smoothing / self.num_classes) * sum_log_probs
        )  # (batch, R)

        weighted = log_loss * rule_sat
        n_matched = rule_sat.sum(dim=0) + 1e-9  # (R,)
        avg_loss_full = weighted.sum(dim=0) / n_matched  # (R,) — per-rule, KHÔNG phụ thuộc ruleset nào
        if self.use_confidence:
            avg_loss_full = avg_loss_full * self.confs

        # ---- Resample K tập luật MỚI từ frozen sampler (MỖI bước) rồi ước
        # lượng MC không chệch của E_{s~pi}[L_rule_penalty(s)]. ----
        masks = self._sample_masks(device).float()  # (K, R)
        self._last_masks = masks.detach()
        mask_count = masks.sum(dim=1).clamp(min=1.0)  # (K,) — ruleset rỗng (nếu có) không gây NaN nhờ clamp
        masked_sum = (masks * avg_loss_full.unsqueeze(0)).sum(dim=1)  # (K,)
        per_ruleset_penalty = masked_sum / mask_count  # (K,) = L_rule_penalty(s_k)/penalty_weight

        mc_estimate = per_ruleset_penalty.mean()  # (1/K) * sum_k L_rule_penalty(s_k)/penalty_weight
        return self.penalty_weight * mc_estimate

    def last_coverage_stats(self) -> dict:
        """Tương tự `VectorizedRulePenalty.last_coverage_stats()`, nhưng
        coverage/active tính trên TOÀN BỘ universe `valid_rules` (không phải
        1 ruleset cố định), cộng thêm `mean_ruleset_size` — kích thước trung
        bình của K ruleset resample gần nhất, để theo dõi xem posterior của
        GFlowNet có ổn định theo thời gian train CNN hay không."""
        n_rules_total = len(self.valid_rules)
        if self._last_rule_sat is None:
            return {
                "n_rules_total": n_rules_total,
                "n_rules_active_this_batch": 0,
                "mean_rule_sat": 0.0,
                "mean_ruleset_size": 0.0,
            }
        rule_sat = self._last_rule_sat
        per_rule_sum = rule_sat.sum(dim=0)
        n_active = int((per_rule_sum >= 1.0).sum().item())
        mean_sat = float(rule_sat.mean().item())
        mean_size = float(self._last_masks.sum(dim=1).mean().item()) if self._last_masks is not None else 0.0
        return {
            "n_rules_total": n_rules_total,
            "n_rules_active_this_batch": n_active,
            "mean_rule_sat": mean_sat,
            "mean_ruleset_size": mean_size,
        }
