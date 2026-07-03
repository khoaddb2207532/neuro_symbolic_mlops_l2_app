"""Chuyển đặc trưng CNN thành mức độ thỏa mãn luật (soft) và tính rule penalty.

LƯU Ý THIẾT KẾ (đã dọn trùng lặp):
  - Trước đây có 2 công thức tính rule-penalty khác nhau (`compute_rule_penalty`
    dạng loop KHÔNG có label smoothing, và `VectorizedRulePenalty` dạng vector
    hoá CÓ label smoothing) — chọn cái nào tuỳ flag `use_vectorized`, khiến độ
    lớn loss có thể khác nhau tuỳ nhánh code dù cùng mục đích. Đã loại bỏ bản
    loop, chỉ giữ `VectorizedRulePenalty` làm cách tính DUY NHẤT dùng khi
    training (nhanh hơn, đã có label smoothing nhất quán với criterion chính).
  - Trước đây `VectorizedRulePenalty` có `update_temperature()` được gọi mỗi
    epoch nhưng `forward()` không hề dùng biến nhiệt độ đó (no-op). Đã bỏ toàn
    bộ state nhiệt độ khỏi class này. Việc khớp luật ở đây dùng so khớp cứng
    (boolean), nhất quán với cách `GPUFastRuleValidator` xác định support/
    confidence của luật — gradient vẫn lan truyền được vào backbone thông qua
    `logits`/`features` trong `log_loss`, mask khớp luật chỉ đóng vai trò
    trọng số, không cần "làm mềm" bằng sigmoid.
  - `BinaryTransformer` (sigmoid, có nhiệt độ) được GIỮ LẠI vì vẫn có công
    dụng thực sự khác: tính "độ khớp luật" liên tục (0-1) để hiển thị cho
    người dùng trong app serving (xem app/inference.py) — mục đích diễn giải
    (explainability), không phải mục đích tính loss khi train.
  - `DynamicBinaryTransformer` (subclass thêm update_temperature) đã bị xoá vì
    không có nơi nào sử dụng — dead code.
"""
import torch
import torch.nn as nn

from src.rules.rule_types import RuleSet


class BinaryTransformer:
    """Tính mức độ thoả mãn luật liên tục (0-1) bằng sigmoid có nhiệt độ.

    Dùng cho mục đích DIỄN GIẢI (hiển thị "độ khớp luật" cho người dùng cuối
    trong app serving), không dùng trong vòng lặp huấn luyện — xem
    VectorizedRulePenalty bên dưới cho phần đó.
    """

    def __init__(self, temperature: float = 10.0):
        self.temperature = temperature

    def transform(self, features: torch.Tensor, rule_set: RuleSet) -> torch.Tensor:
        N = features.size(0)
        if len(rule_set) == 0:
            return torch.empty((N, 0), dtype=torch.float32, device=features.device)
        binary_list = []
        for rule in rule_set.rules:
            sat = torch.ones(N, device=features.device)
            for cond in rule.conditions:
                col = features[:, cond.feature_index]
                if cond.operator == "<=":
                    sat = sat * torch.sigmoid(self.temperature * (cond.threshold - col))
                else:
                    sat = sat * torch.sigmoid(self.temperature * (col - cond.threshold))
            binary_list.append(sat.unsqueeze(1))
        return torch.cat(binary_list, dim=1)


class VectorizedRulePenalty(nn.Module):
    """Rule-penalty vector hoá đầy đủ trên GPU — cách tính DUY NHẤT dùng khi
    huấn luyện (thay thế công thức loop cũ đã bị xoá).

    Khớp luật bằng so khớp cứng (boolean, giống GPUFastRuleValidator), có
    Label Smoothing để nhất quán với CrossEntropyLoss chính của mô hình.
    """

    def __init__(
        self,
        rule_set,
        penalty_weight: float = 0.1,
        use_confidence: bool = True,
        smoothing: float = 0.1,
        num_classes: int = 12,
    ):
        super().__init__()
        self.rule_set = rule_set
        self.penalty_weight = penalty_weight
        self.use_confidence = use_confidence
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        R = len(self.rule_set.rules)
        if R == 0:
            return torch.tensor(0.0, device=features.device)
        device = features.device
        batch_size = features.size(0)
        max_conds = max(len(r.conditions) for r in self.rule_set.rules)

        feat_idx = torch.zeros(R, max_conds, dtype=torch.long, device=device)
        thresholds = torch.zeros(R, max_conds, dtype=torch.float, device=device)
        ops = torch.zeros(R, max_conds, dtype=torch.bool, device=device)
        valid_m = torch.zeros(R, max_conds, dtype=torch.bool, device=device)
        targets = torch.tensor([r.target_class for r in self.rule_set.rules], device=device)
        confs = torch.tensor([r.confidence for r in self.rule_set.rules], device=device)

        for i, rule in enumerate(self.rule_set.rules):
            for j, cond in enumerate(rule.conditions):
                feat_idx[i, j] = cond.feature_index
                thresholds[i, j] = cond.threshold
                ops[i, j] = cond.operator == ">"
                valid_m[i, j] = True

        sel = features[:, feat_idx]
        cond_ok = ((sel <= thresholds) & ~ops) | ((sel > thresholds) & ops)
        cond_ok = cond_ok | ~valid_m
        rule_sat = cond_ok.all(dim=-1).float()

        log_probs = torch.log_softmax(logits, dim=1)
        tgt_log_probs = log_probs.gather(1, targets.unsqueeze(0).expand(batch_size, -1))
        sum_log_probs = log_probs.sum(dim=1, keepdim=True).expand(-1, R)

        log_loss = -(
            (1.0 - self.smoothing) * tgt_log_probs + (self.smoothing / self.num_classes) * sum_log_probs
        )

        weighted = log_loss * rule_sat
        n_matched = rule_sat.sum(dim=0) + 1e-9
        avg_loss = weighted.sum(dim=0) / n_matched
        if self.use_confidence:
            avg_loss = avg_loss * confs
        return self.penalty_weight * avg_loss.mean()
