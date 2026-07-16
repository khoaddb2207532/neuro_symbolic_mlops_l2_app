"""Lọc luật dựa trên support/precision, vector hoá trên GPU.

Chỉ còn 1 phương thức lọc luật: `validate_and_build_tensors()`.

Cập nhật theo thuật toán "Lọc và Chuẩn bị Tập Ứng viên Luật Tối ưu"
(filter_and_prepare_candidates): toàn bộ việc đánh giá lại luật — gán lại
nhãn theo đa số, tính base rate, lọc theo coverage/precision — đều thực
hiện trên TOÀN BỘ `training_data`, đúng như mô tả gốc:

  1. GÁN LẠI NHÃN (re-label / re-measure trên training_data): luật trích từ
     một cây đơn lẻ trong Random Forest thường bị phân mảnh vì mỗi cây chỉ
     được huấn luyện trên một mẫu bootstrap của training_data, nên
     `target_class` gốc lấy từ cây không còn đáng tin. Ta đánh giá lại điều
     kiện của luật trên TOÀN BỘ training_data (không chỉ bootstrap riêng
     của cây đó) và gán `predicted_class = argmax` trên phân phối lớp thực
     tế của các mẫu bị luật đó phủ.
  2. BASE RATE tính trên training_data: base_rates[y] = tỷ lệ mẫu lớp y
     trên toàn bộ training_data.
  3. LỌC THEO NGƯỠNG (trên training_data):
     - Ngưỡng 1 (coverage): loại luật có frequency < theta_cov (`min_supp`).
     - Ngưỡng 2 (base rate): loại luật có precision <= base_rates[predicted_class]
       — luật không "thắng" được việc đoán mù theo phân phối lớp thì vô dụng,
       không mang thêm thông tin gì cho GFlowNet xử lý tiếp.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.rules.rule_types import Rule, RuleSet


class RuleValidator:
    def __init__(
        self,
        min_supp: float = 0.01,
        rule_batch_size: int = 2000,
        data_batch_size: int = 20000,
    ):
        self.min_supp = min_supp
        self.rule_batch_size = rule_batch_size
        self.data_batch_size = data_batch_size

    @staticmethod
    def _build_rule_tensors(batch_rules: List[Rule], device):
        """Chỉ dựng tensor mô tả ĐIỀU KIỆN của luật (feature/threshold/toán tử).
        Không dùng target_class gốc của cây ở đây — nhãn được gán lại
        (re-label) dựa trên training_data thực tế, xem `validate_and_build_tensors`.
        """
        B = len(batch_rules)
        max_conds = max((len(r.conditions) for r in batch_rules), default=1)
        feat_idx = torch.zeros(B, max_conds, dtype=torch.long, device=device)
        thresholds = torch.zeros(B, max_conds, dtype=torch.float32, device=device)
        ops = torch.zeros(B, max_conds, dtype=torch.bool, device=device)
        valid_m = torch.zeros(B, max_conds, dtype=torch.bool, device=device)
        for j, rule in enumerate(batch_rules):
            for k, cond in enumerate(rule.conditions):
                feat_idx[j, k] = cond.feature_index
                thresholds[j, k] = cond.threshold
                ops[j, k] = cond.operator == ">"
                valid_m[j, k] = True
        return feat_idx, thresholds, ops, valid_m

    def validate_and_build_tensors(
        self,
        rule_set: RuleSet,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        store_device: torch.device = torch.device("cpu"),
        num_classes: Optional[int] = None,
    ) -> Tuple[RuleSet, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Một lần quét duy nhất qua đặc trưng của TOÀN BỘ training_data: vừa
        tính độ phủ (cover) của mỗi luật, vừa gán lại nhãn theo đa số thực tế
        (re-label), vừa lọc theo min_supp và base rate, vừa trả về
        cover/correct/rule_len CHỈ cho các rule được giữ — không tính lại
        lần 2 trên đặc trưng.

        `train_features`/`train_labels` ở đây chính là training_data trong
        mô tả thuật toán (toàn bộ dữ liệu huấn luyện, KHÔNG phải một
        bootstrap sample riêng của từng cây).

        Nhãn `target_class` và `confidence` của rule được GHI ĐÈ theo kết
        quả đánh giá lại trên training_data.
        """
        device = train_features.device
        N = train_features.size(0)
        train_labels = train_labels.view(-1).long()

        # ==========================================
        # BASE RATE: tỷ lệ phân phối tự nhiên của mỗi lớp trên toàn bộ training_data
        # ==========================================
        if num_classes is None:
            num_classes = int(train_labels.max().item()) + 1
        class_totals = torch.bincount(train_labels, minlength=num_classes).float()
        base_rates = class_totals / N  # P_D của mỗi lớp
        one_hot_labels = F.one_hot(train_labels, num_classes).float()  # (N, num_classes)

        filtered_rules: List[Rule] = []
        cover_chunks: List[torch.Tensor] = []
        correct_chunks: List[torch.Tensor] = []
        rule_len_chunks: List[torch.Tensor] = []

        for i in tqdm(range(0, len(rule_set.rules), self.rule_batch_size), desc="Validate + build tensors"):
            batch_rules = rule_set.rules[i : i + self.rule_batch_size]
            B = len(batch_rules)
            feat_idx, thresholds, ops, valid_m = self._build_rule_tensors(batch_rules, device)

            batch_cover = torch.zeros((B, N), dtype=torch.bool, device=device)

            for d_start in range(0, N, self.data_batch_size):
                d_end = min(N, d_start + self.data_batch_size)
                feat_chunk = train_features[d_start:d_end]

                sel = feat_chunk[:, feat_idx]
                cond_ok = ((sel <= thresholds) & ~ops) | ((sel > thresholds) & ops)
                cond_ok = cond_ok | ~valid_m
                rule_mask = cond_ok.all(dim=-1)  # (n_chunk, B)

                batch_cover[:, d_start:d_end] = rule_mask.T

            # ==========================================================
            # CÔNG VIỆC 1: ĐÁNH GIÁ LẠI TRÊN TOÀN BỘ TRAINING_DATA (RE-MEASURE)
            # ==========================================================
            # Phân phối lớp của các mẫu bị mỗi luật phủ trên training_data,
            # tính một lần bằng matmul thay vì quét lại đặc trưng:
            # class_counts[j, c] = số mẫu bị luật j phủ và có nhãn thật là c.
            class_counts = batch_cover.float() @ one_hot_labels  # (B, num_classes)
            total_supp = class_counts.sum(dim=1)  # n_covered của mỗi luật
            total_corr, predicted_class = class_counts.max(dim=1)  # gán lại nhãn = lớp đa số

            supp_ratio = total_supp / N  # frequency / coverage
            precision = torch.zeros(B, device=device)
            valid_mask = total_supp > 0  # tránh chia cho 0 (luật không phủ mẫu nào trên training_data)
            precision[valid_mask] = total_corr[valid_mask] / total_supp[valid_mask]

            # ==========================================================
            # CÔNG VIỆC 3: LỌC CỨNG THEO NGƯỠNG (THRESHOLDING, trên training_data)
            # ==========================================================
            # Ngưỡng 1: loại luật overfitting có độ phủ quá thấp (frequency < theta_cov)
            # Ngưỡng 2: loại luật vô dụng nếu precision không CAO HƠN base rate
            # của lớp mà nó dự đoán (precision <= base_rate => loại)
            keep = valid_mask & (supp_ratio >= self.min_supp) & (precision > base_rates[predicted_class])
            keep_idx = torch.where(keep)[0]

            batch_correct = batch_cover & (train_labels.unsqueeze(0) == predicted_class.unsqueeze(1))

            for idx in keep_idx.cpu().tolist():
                rule = batch_rules[idx]
                rule.target_class = int(predicted_class[idx].item())  # gán lại nhãn theo đa số
                rule.confidence = precision[idx].item()
                filtered_rules.append(rule)

            if keep_idx.numel() > 0:
                cover_chunks.append(batch_cover[keep_idx].to(store_device))
                correct_chunks.append(batch_correct[keep_idx].to(store_device))
                rule_len_chunks.append(valid_m[keep_idx].sum(dim=-1).to(store_device).float())

            del batch_cover, batch_correct  # giải phóng ngay, tránh giữ VRAM qua các batch

        cover = torch.cat(cover_chunks, dim=0) if cover_chunks else torch.zeros((0, N), dtype=torch.bool)
        correct = torch.cat(correct_chunks, dim=0) if correct_chunks else torch.zeros((0, N), dtype=torch.bool)
        rule_len = torch.cat(rule_len_chunks, dim=0) if rule_len_chunks else torch.zeros(0)

        return RuleSet(rules=filtered_rules), cover, correct, rule_len