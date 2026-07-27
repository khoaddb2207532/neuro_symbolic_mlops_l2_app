"""Chuyển đặc trưng CNN thành mức độ thỏa mãn luật (soft) và tính rule penalty.

LỊCH SỬ THIẾT KẾ (đọc để hiểu vì sao có 2 lần đổi hướng về "nhiệt độ"):
  - Từng có 2 công thức tính rule-penalty khác nhau (loop không label smoothing
    vs vectorized có label smoothing) — đã gộp chỉ còn 1 (VectorizedRulePenalty).
  - Sau đó, `VectorizedRulePenalty` dùng so khớp CỨNG (boolean) và từng có một
    `update_temperature()` KHÔNG được `forward()` sử dụng (dead code) — đã bị
    xoá tạm thời.
  - NÂNG CẤP LẦN NÀY: đưa khớp MỀM (sigmoid có nhiệt độ, ủ dần cứng lên theo
    epoch) trở lại — nhưng lần này `forward()` THỰC SỰ dùng nhiệt độ. Lý do:
    với khớp cứng, `rule_sat` là phép so sánh `<=`/`>` không khả vi, nên
    gradient chỉ chảy được vào `logits` (qua log_loss) cho các sample đã sẵn
    nằm trong vùng thoả luật — KHÔNG kéo được `features` lại gần vùng thoả
    luật hơn. Khớp mềm bằng sigmoid làm `rule_sat` trở thành hàm khả vi theo
    `features`, nên rule-penalty giờ thực sự regularize được không gian biểu
    diễn (representation), không chỉ quyết định phân loại ở lớp cuối. Nhiệt
    độ ủ dần thấp→cao qua các epoch để "mềm" lúc đầu (dễ tối ưu, gradient
    mượt) rồi "cứng" dần về cuối (khớp luật gần đúng nghĩa boolean thật).
  - `GPUFastRuleValidator.validate()` (dùng ở stage3 để tính support/confidence
    của luật) VẪN dùng so khớp CỨNG — đây là lựa chọn có chủ đích khác: thống
    kê support/confidence cần là số đếm chính xác (đúng nghĩa luật), còn
    penalty lúc train thì cần khả vi. Hai nơi dùng 2 kiểu khớp cho 2 mục đích
    khác nhau, không phải xung đột.
  - `BinaryTransformer` (sigmoid, có nhiệt độ) vẫn dùng cho mục đích DIỄN GIẢI
    ở app serving (nhiệt độ cố định, không ủ dần) — tách biệt với
    `VectorizedRulePenalty` (nhiệt độ ủ dần theo epoch khi train).
"""
from typing import Optional

import torch
import torch.nn as nn

from src.rules.rule_types import RuleSet
from src.rules.temperature import geometric_temperature


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
    """Rule-penalty vector hoá đầy đủ trên GPU, khớp luật MỀM (soft matching)
    bằng sigmoid có nhiệt độ ủ dần (anneal) từ `initial_temp` (mềm) xuống
    `final_temp` (gần cứng) theo epoch — xem lý do ở docstring đầu file.

    Có Label Smoothing để nhất quán với CrossEntropyLoss chính của mô hình.
    """

    def __init__(
        self,
        rule_set,
        penalty_weight: float = 0.1,
        use_confidence: bool = True,
        smoothing: float = 0.1,
        num_classes: int = 12,
        initial_temp: float = 2.0,
        final_temp: float = 15.0,
        temp_warmup_epochs: int = 2,
        temp_anneal_epochs: int = 10,
        active_threshold: float = 1.0,
    ):
        super().__init__()
        self.rule_set = rule_set
        self.penalty_weight = penalty_weight
        self.use_confidence = use_confidence
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.temp_warmup_epochs = temp_warmup_epochs
        self.temp_anneal_epochs = temp_anneal_epochs
        # Ngưỡng (trên tổng rule_sat của cả batch) để coi 1 luật là "active"
        # trong batch đó — xem last_coverage_stats(). Mặc định 1.0 tương
        # đương "ít nhất khoảng 1 sample khớp gần như hoàn toàn" theo thang
        # soft-matching liên tục.
        self.active_threshold = active_threshold
        # Buffer (không phải Parameter): nhiệt độ được lịch trình hoá theo
        # epoch bởi trainer, không học qua backprop.
        self.register_buffer("_temperature", torch.tensor(float(initial_temp)))
        # Không phải buffer/parameter — chỉ là chỗ lưu tạm rule_sat của lần
        # forward() gần nhất để đọc cho mục đích observability (không tham
        # gia checkpoint, không cần đồng bộ device đặc biệt vì luôn được ghi
        # đè ngay sau forward() tiếp theo).
        self._last_rule_sat: Optional[torch.Tensor] = None

    def update_temperature(self, epoch: int) -> None:
        """Ủ theo lịch cố định, không phụ thuộc giới hạn ``num_epochs``."""
        new_temp = geometric_temperature(
            epoch,
            self.initial_temp,
            self.final_temp,
            self.temp_warmup_epochs,
            self.temp_anneal_epochs,
        )
        self._temperature.fill_(new_temp)

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        R = len(self.rule_set.rules)
        if R == 0:
            self._last_rule_sat = None
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

        # ---- Khớp luật MỀM (thay cho so khớp cứng trước đây) ----
        sel = features[:, feat_idx]  # (batch, R, max_conds)
        temp = self._temperature
        sat_le = torch.sigmoid(temp * (thresholds - sel))  # điều kiện "<="
        sat_gt = torch.sigmoid(temp * (sel - thresholds))  # điều kiện ">"
        cond_sat = torch.where(ops, sat_gt, sat_le)
        # Điều kiện "không tồn tại" (do padding max_conds) không được ảnh
        # hưởng tới tích AND-mềm -> gán 1.0 (trung tính).
        cond_sat = torch.where(valid_m, cond_sat, torch.ones_like(cond_sat))
        rule_sat = cond_sat.prod(dim=-1)  # AND mềm = tích các điều kiện, (batch, R) trong (0,1)
        # Lưu lại (detach, không giữ graph) để last_coverage_stats() đọc được
        # — mục đích observability, KHÔNG dùng lại giá trị này khi backward().
        self._last_rule_sat = rule_sat.detach()

        log_probs = torch.log_softmax(logits, dim=1)
        tgt_log_probs = log_probs.gather(1, targets.unsqueeze(0).expand(batch_size, -1))
        sum_log_probs = log_probs.sum(dim=1, keepdim=True).expand(-1, R)

        log_loss = -(
            (1.0 - self.smoothing) * tgt_log_probs + (self.smoothing / self.num_classes) * sum_log_probs
        )

        weighted = log_loss * rule_sat
        n_matched = rule_sat.sum(dim=0) + 1e-9  # "số sample khớp" giờ là tổng liên tục, không phải đếm nguyên
        avg_loss = weighted.sum(dim=0) / n_matched
        if self.use_confidence:
            avg_loss = avg_loss * confs
        return self.penalty_weight * avg_loss.mean()

    def last_coverage_stats(self) -> dict:
        """Thống kê coverage của lần forward() GẦN NHẤT — gọi sau khi đã gọi
        forward() ít nhất 1 lần trong batch hiện tại (xem
        src/training/trainer.py::train_one_epoch()).

        Trả về:
          - n_rules_total: tổng số luật trong rule_set (không đổi theo batch)
          - n_rules_active_this_batch: số luật có tổng rule_sat trên cả batch
            vượt `active_threshold` — tức thực sự đóng góp gradient đáng kể
            trong batch này (không phải chỉ "được GFlowNet chọn" mà không
            bao giờ khớp sample nào)
          - mean_rule_sat: mức độ khớp trung bình (0-1) trên toàn bộ (batch, R)
        """
        n_rules_total = len(self.rule_set.rules)
        if self._last_rule_sat is None:
            return {
                "n_rules_total": n_rules_total,
                "n_rules_active_this_batch": 0,
                "mean_rule_sat": 0.0,
            }
        rule_sat = self._last_rule_sat  # (batch, R), đã detach
        per_rule_sum = rule_sat.sum(dim=0)  # (R,)
        n_active = int((per_rule_sum >= self.active_threshold).sum().item())
        mean_sat = float(rule_sat.mean().item())
        return {
            "n_rules_total": n_rules_total,
            "n_rules_active_this_batch": n_active,
            "mean_rule_sat": mean_sat,
        }
