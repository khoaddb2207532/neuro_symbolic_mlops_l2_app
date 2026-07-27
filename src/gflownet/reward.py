"""Reward chọn ruleset cho mục tiêu regularize CNN.

Score = w_acc * MacroAccuracy
      + w_cov * CorrectCoverage
      - w_wrong * WrongCoverage
      - w_conflict * ConfidenceWeightedConflict.

Ruleset dự đoán bằng weighted voting theo confidence của các luật được chọn.
"""
import torch


class RuleSetReward:
    def __init__(self, cover: torch.Tensor, correct: torch.Tensor,
                 rule_len: torch.Tensor, max_rules: int, targets: torch.Tensor,
                 labels: torch.Tensor, confidences: torch.Tensor,
                 w_acc=1.0, w_cov=0.5, w_wrong=0.75,
                 w_conflict=0.1, beta=3.0):
        """
        cover:    (n_rules, n_val) bool - luật i có match mẫu j không
        correct:  (n_rules, n_val) bool - luật i match VÀ dự đoán đúng nhãn mẫu j
        rule_len: (n_rules,) - số điều kiện trong luật (giữ lại để dùng ngoài
                  reward, ví dụ đo độ phức tạp mô tả cho phần diễn giải, không
                  còn là 1 số hạng trong score() nữa — xem ghi chú bên dưới)
        targets:  (n_rules,) long - target_class của từng luật, dùng để tách
                  "trùng lặp vô hại" (2 luật cùng phủ 1 vùng, CÙNG target) khỏi
                  "xung đột thật sự" (2 luật cùng phủ 1 vùng, KHÁC target).

        THIẾT KẾ (đã đổi so với bản đầu): reward này dùng để CHỌN luật phục vụ
        MỘT mục đích duy nhất — làm regularizer tốt cho CNN ở stage5, không
        phải để tối ưu tính "gọn/dễ đọc" của tập luật (đó là việc của thống kê
        mô tả ở bước phân tích luật riêng, không phải của GFlowNet). Vì vậy:
          - ĐÃ BỎ complexity (phạt số lượng luật): nhiều luật đúng, không xung
            đột không hề xấu cho regularization — ngược lại còn cho nhiều
            sample hơn nhận được tín hiệu phụ hữu ích.
          - ĐÃ BỎ phần "trùng lặp cùng target" khỏi phạt: 2 luật cùng đúng,
            cùng phủ 1 vùng không gây hại cho CNN (giống hiệu ứng đồng thuận
            trong ensemble) — chỉ giữ lại phạt cho phần THỰC SỰ gây hại:
            2 luật cùng phủ 1 vùng nhưng khác target (xung đột), đúng vấn đề
            đã xử lý ở RulePenaltyLoss (soft-target per-sample).

        SỬA LẦN 2 — conflict đo THEO MẪU, không đo THEO CẶP LUẬT: bản trước
        tính `redundancy_conflict` bằng trung bình Jaccard trên các CẶP luật
        xung đột, chia cho `n_pairs = n_selected*(n_selected-1)`. Vấn đề: số
        hạng này bị PHA LOÃNG khi ruleset lớn dần (n_pairs tăng bậc hai theo
        số luật), nên GFlowNet có thể "nhồi" thêm luật rác để pha loãng phạt
        xung đột mà gần như không tốn gì — đây chính là lý do reward không
        thể giảm sâu cho ruleset thực sự tệ (xem phân tích trong hội thoại).

        Cách sửa dựa theo cách đo overlap trong các nghiên cứu về rule-set
        learning (vd Lakkaraju et al., "Interpretable Decision Sets", KDD
        2016; và "Learning Interpretable Decision Rule Sets: A Submodular
        Optimization Approach", NeurIPS 2021): đo overlap là TỈ LỆ MẪU bị
        phủ bởi nhiều hơn 1 luật, không phải trung bình theo cặp luật. Ở đây
        cụ thể hoá thành: `conflict_ratio` = tỉ lệ mẫu (trên tổng n_val) bị
        phủ đồng thời bởi các luật ĐƯỢC CHỌN thuộc >= 2 target khác nhau.
        Số hạng này chia cho `n_val` (CỐ ĐỊNH, không đổi theo ruleset), nên
        KHÔNG bị pha loãng khi thêm luật — ngược lại, thêm luật xung đột
        thực sự sẽ làm `conflict_ratio` tăng lên đúng theo số mẫu bị ảnh
        hưởng, phản ánh đúng mức độ "hỏng" của ruleset.
        """
        self.cover = cover.float()
        self.correct = correct.float()
        self.rule_len = rule_len.float()
        self.n_val = cover.shape[1]
        self.max_rules = max_rules
        self.w_acc = w_acc
        self.w_cov = w_cov
        self.w_wrong = w_wrong
        self.w_conflict = w_conflict
        self.beta = beta

        # ma trận overlap giữa các luật (Jaccard) — CHỈ dùng cho thống kê mô
        # tả bên ngoài (evaluate_run báo cáo độ trùng lặp tổng quát, không
        # phải component GFlowNet đang tối ưu) — không còn dùng bên trong
        # score() nữa (xem conflict_ratio ở components()/score() bên dưới).
        inter = cover.float() @ cover.float().T
        card = cover.float().sum(-1, keepdim=True)
        union = card + card.T - inter
        self.jaccard = inter / union.clamp(min=1e-8)   # (n_rules, n_rules) — toàn bộ overlap

        targets = targets.to(cover.device)
        self.targets = targets
        self.labels = labels.to(cover.device).long()
        self.confidences = confidences.to(cover.device).float()
        max_target = int(targets.max().item()) if targets.numel() > 0 else -1
        max_label = int(self.labels.max().item()) if self.labels.numel() > 0 else -1
        self.num_classes = max(max_target, max_label) + 1
        # class_masks[c, i] = 1 nếu luật i có target = c — dùng để gộp các
        # luật ĐƯỢC CHỌN theo từng target riêng biệt (xem components()).
        self.class_masks = torch.zeros(self.num_classes, cover.shape[0], device=cover.device)
        if self.num_classes > 0:
            self.class_masks.scatter_(0, targets.unsqueeze(0), 1.0)

    def components(self, s: torch.Tensor) -> dict:
        """Tách riêng từng thành phần của reward — dùng chung bởi `score()`
        VÀ bởi `evaluation.py::evaluate_run()` để tránh 2 nơi tính công thức
        khác nhau (trước đây evaluation.py tự tính lại `redundancy_conflict`
        một cách độc lập, dễ lệch khỏi công thức thật trong score()).

        s: (B, n_rules) nhị phân. Trả về dict các tensor (B,).
        """
        if self.num_classes == 0:
            zeros = torch.zeros(s.shape[0], device=s.device)
            return {
                "n_selected": s.sum(-1),
                "coverage": zeros,
                "accuracy": zeros,
                "macro_accuracy": zeros,
                "correct_coverage": zeros,
                "wrong_coverage": zeros,
                "conflict_ratio": zeros,
            }

        # votes[b,c,j] là tổng confidence của các luật target=c trong ruleset b
        # cùng phủ mẫu validation j.
        weighted_selected = s * self.confidences.unsqueeze(0)                # (B, R)
        selected_by_class = (
            weighted_selected.unsqueeze(1) * self.class_masks.unsqueeze(0)
        )                                                                    # (B, C, R)
        votes = torch.einsum(
            "bcr,rj->bcj", selected_by_class, self.cover
        )                                                                    # (B, C, N)
        total_votes = votes.sum(dim=1)                                       # (B, N)
        covered = total_votes > 0
        predictions = votes.argmax(dim=1)                                    # (B, N)
        labels = self.labels.unsqueeze(0)
        correct_predictions = covered & (predictions == labels)
        wrong_predictions = covered & (predictions != labels)

        # MacroAccuracy được tính có điều kiện trên phần đã phủ của từng lớp;
        # Correct/WrongCoverage dùng toàn bộ số mẫu của lớp làm mẫu số.
        per_class_accuracy = []
        per_class_correct_coverage = []
        per_class_wrong_coverage = []
        for class_idx in range(self.num_classes):
            class_mask = self.labels == class_idx
            if not class_mask.any():
                continue
            class_count = class_mask.sum().float()
            class_covered = covered[:, class_mask]
            class_correct = correct_predictions[:, class_mask].float().sum(-1)
            class_wrong = wrong_predictions[:, class_mask].float().sum(-1)
            per_class_accuracy.append(
                class_correct / class_covered.float().sum(-1).clamp(min=1)
            )
            per_class_correct_coverage.append(class_correct / class_count)
            per_class_wrong_coverage.append(class_wrong / class_count)

        macro_accuracy = torch.stack(per_class_accuracy, dim=-1).mean(-1)
        correct_coverage = torch.stack(
            per_class_correct_coverage, dim=-1
        ).mean(-1)
        wrong_coverage = torch.stack(
            per_class_wrong_coverage, dim=-1
        ).mean(-1)
        coverage = correct_coverage + wrong_coverage

        # Conflict liên tục có trọng số confidence: phần vote không thuộc lớp
        # thắng, chuẩn hóa theo tổng vote. Bằng 0 khi các luật hoàn toàn đồng thuận.
        winning_votes = votes.max(dim=1).values
        per_sample_conflict = (
            (total_votes - winning_votes) / total_votes.clamp(min=1e-8)
        )
        conflict_ratio = per_sample_conflict.mean(-1)

        return {
            "n_selected": s.sum(-1),
            "coverage": coverage,
            "accuracy": macro_accuracy,
            "macro_accuracy": macro_accuracy,
            "correct_coverage": correct_coverage,
            "wrong_coverage": wrong_coverage,
            "conflict_ratio": conflict_ratio,
        }

    def score(self, s: torch.Tensor) -> torch.Tensor:
        # s: (B, n_rules) nhị phân
        c = self.components(s)
        return (
            self.w_acc * c["macro_accuracy"]
            + self.w_cov * c["correct_coverage"]
            - self.w_wrong * c["wrong_coverage"]
            - self.w_conflict * c["conflict_ratio"]
        )

    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        raw = self.score(s)                       # có thể âm
        return torch.exp(self.beta * raw)          # đảm bảo R > 0
