"""Khối luật (ConceptRuleHead) — cộng thêm logit từ các luật concept vào CBM.

Ý tưởng: thay vì chỉ dùng luật làm regularizer lúc huấn luyện (``VectorizedRulePenalty``),
khối này đưa luật VÀO đường dự đoán. Với luật r: ``(l_1 ∧ ... ∧ l_m) => c_r``,

    rule_sat_r(p)   = Π_i sigmoid(T · (p_{k_i} - 0.5))   (literal dương)
                      Π_i sigmoid(T · (0.5 - p_{k_i}))   (literal âm)     ∈ (0, 1)
    logits_extra[c] = Σ_r  w_r · rule_sat_r(p) · [c_r == c]

    logits_final   = label_predictor(p) + logits_extra

``w_r`` là tham số học được, KHỞI TẠO = 0. Hệ quả:
  - Ngay khi gắn khối, mô hình trùng đúng CBM đã train (không phá checkpoint).
  - Gradient chảy vào w_r và qua rule_sat vào concept layer, nên luật thực sự
    điều chỉnh biểu diễn concept, không chỉ là một bảng tra cứu cố định.
  - Mỗi dự đoán có thể giải thích được: các luật có rule_sat cao và w_r > 0 là
    lý do của lớp được chọn.

``RuleAugmentedCBM`` là CBM với khối này. Nó kế thừa toàn bộ giao diện của
``ConceptBottleneckModel`` (trả cùng bộ 4 giá trị, có ``trainable_param_groups``),
nên ``train_cbm`` và ``evaluate_model_performance`` dùng được mà không sửa.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cbm import ConceptBottleneckModel
from src.rules.rule_types import RuleSet


class ConceptRuleHead(nn.Module):
    def __init__(
        self,
        rule_set: RuleSet,
        num_classes: int,
        temperature: float = 10.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature phải > 0.")
        self.num_classes = num_classes
        R = len(rule_set.rules)
        self.n_rules = R
        max_conds = max((len(r.conditions) for r in rule_set.rules), default=1)

        feat_idx = torch.zeros(R, max_conds, dtype=torch.long)
        thresholds = torch.zeros(R, max_conds, dtype=torch.float)
        ops_gt = torch.zeros(R, max_conds, dtype=torch.bool)
        valid = torch.zeros(R, max_conds, dtype=torch.bool)
        targets = torch.zeros(R, dtype=torch.long)
        for i, rule in enumerate(rule_set.rules):
            if not 0 <= rule.target_class < num_classes:
                raise ValueError(
                    f"Luật {i} có target_class={rule.target_class} ngoài [0, {num_classes})."
                )
            targets[i] = rule.target_class
            for j, cond in enumerate(rule.conditions):
                feat_idx[i, j] = cond.feature_index
                thresholds[i, j] = cond.threshold
                ops_gt[i, j] = cond.operator == ">"
                valid[i, j] = True

        self.register_buffer("feat_idx", feat_idx)
        self.register_buffer("thresholds", thresholds)
        self.register_buffer("ops_gt", ops_gt)
        self.register_buffer("valid", valid)
        self.register_buffer("targets", targets)
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.rule_weight = nn.Parameter(torch.zeros(R))

    def rule_satisfaction(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """(B, R) — mức thoả luật mềm trong (0, 1), khả vi theo concept_probs."""
        B = concept_probs.shape[0]
        if self.n_rules == 0:
            return concept_probs.new_zeros(B, 0)
        sel = concept_probs[:, self.feat_idx]                       # (B, R, M)
        sat_le = torch.sigmoid(self.temperature * (self.thresholds - sel))
        sat_gt = torch.sigmoid(self.temperature * (sel - self.thresholds))
        cond = torch.where(self.ops_gt, sat_gt, sat_le)
        cond = torch.where(self.valid, cond, torch.ones_like(cond))  # padding trung tính
        return cond.prod(dim=-1)

    def forward(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """(B, num_classes) — logit cộng thêm từ các luật."""
        B = concept_probs.shape[0]
        if self.n_rules == 0:
            return concept_probs.new_zeros(B, self.num_classes)
        sat = self.rule_satisfaction(concept_probs)                  # (B, R)
        onehot = F.one_hot(self.targets, self.num_classes).to(sat.dtype)  # (R, C)
        return (sat * self.rule_weight) @ onehot

    def explain(self, concept_probs: torch.Tensor, top_k: int = 3) -> list:
        """Với mỗi mẫu, liệt kê top-k luật đóng góp nhiều nhất vào lớp dự đoán."""
        with torch.no_grad():
            sat = self.rule_satisfaction(concept_probs)
            contrib = sat * self.rule_weight                          # (B, R)
        out = []
        for b in range(contrib.shape[0]):
            vals, idx = torch.topk(contrib[b], min(top_k, self.n_rules)) if self.n_rules else (
                torch.empty(0), torch.empty(0, dtype=torch.long))
            out.append([{"rule_index": int(i), "contribution": float(v),
                         "target_class": int(self.targets[i])} for v, i in zip(vals, idx)])
        return out


class RuleAugmentedCBM(ConceptBottleneckModel):
    """CBM + khối luật. Logit cuối = head(p) + ConceptRuleHead(p)."""

    def __init__(
        self,
        rule_head: ConceptRuleHead,
        rule_lr: Optional[float] = None,
        **cbm_kwargs,
    ) -> None:
        super().__init__(**cbm_kwargs)
        if rule_head.num_classes != self.num_classes:
            raise ValueError(
                f"rule_head.num_classes={rule_head.num_classes} khác CBM num_classes={self.num_classes}."
            )
        self.rule_head = rule_head
        self.rule_lr = rule_lr

    def forward_with_concept_logits(self, x: torch.Tensor):
        raw_features = self.backbone(x)
        concept_logits = self.concept_layer[0](raw_features)
        concept_probs = self.concept_layer[1](concept_logits)
        logits = self.label_predictor(concept_probs) + self.rule_head(concept_probs)
        return logits, concept_probs, raw_features, concept_logits

    def trainable_param_groups(self, lr_head: float, lr_backbone: float):
        """Giữ 2 nhóm như CBM; khối luật có nhóm riêng với ``rule_lr`` (nếu đặt).

        Khối luật khởi tạo bằng 0 nên cần LR đủ lớn để w_r rời khỏi 0 trong số bước
        huấn luyện có sẵn; dùng chung LR head rất chậm.
        """
        groups = super().trainable_param_groups(lr_head=lr_head, lr_backbone=lr_backbone)
        rule_params = [p for p in self.rule_head.parameters() if p.requires_grad]
        if rule_params:
            groups.append({
                "params": rule_params,
                "lr": self.rule_lr if self.rule_lr is not None else lr_head,
                "name": "rule",
            })
        return groups
