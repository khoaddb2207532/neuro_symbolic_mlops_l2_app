import torch


class RuleSetReward:
    def __init__(self, cover: torch.Tensor, correct: torch.Tensor,
                 rule_len: torch.Tensor, max_rules: int, targets: torch.Tensor,
                 w_acc=1.0, w_cov=0.5, w_conflict=0.5, beta=3.0):
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
        """
        self.cover = cover.float()
        self.correct = correct.float()
        self.rule_len = rule_len.float()
        self.n_val = cover.shape[1]
        self.max_rules = max_rules
        self.w_acc, self.w_cov, self.w_conflict = w_acc, w_cov, w_conflict
        self.beta = beta

        # ma trận overlap giữa các luật (Jaccard) — tính 1 lần, dùng cho cả
        # reward (chỉ phần "conflict") lẫn thống kê mô tả bên ngoài (evaluate_run
        # vẫn dùng self.jaccard đầy đủ để báo cáo độ trùng lặp tổng quát, tách
        # biệt với việc GFlowNet có bị phạt vì nó hay không).
        inter = cover.float() @ cover.float().T
        card = cover.float().sum(-1, keepdim=True)
        union = card + card.T - inter
        self.jaccard = inter / union.clamp(min=1e-8)   # (n_rules, n_rules) — toàn bộ overlap

        targets = targets.to(cover.device)
        same_target = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()  # (n_rules, n_rules)
        # Chỉ phần KHÁC target mới là "xung đột" — đây là số hạng DUY NHẤT
        # còn lại trong score() liên quan tới overlap giữa các luật.
        self.jaccard_conflict = self.jaccard * (1.0 - same_target)

    def score(self, s: torch.Tensor) -> torch.Tensor:
        # s: (B, n_rules) nhị phân
        n_selected = s.sum(-1).clamp(min=1)

        covered = (s @ self.cover) > 0                       # (B, n_val)
        correct_covered = (s @ self.correct) > 0
        coverage = covered.float().mean(-1)
        # accuracy chỉ tính trên phần được phủ
        accuracy = (correct_covered.float().sum(-1)) / covered.float().sum(-1).clamp(min=1)

        # Xung đột trung bình giữa các cặp luật đã chọn (chỉ tính cặp KHÁC
        # target — trùng lặp CÙNG target không bị phạt, xem docstring ở trên).
        pair_conflict = (s.unsqueeze(1) * s.unsqueeze(2) * self.jaccard_conflict).sum((-1, -2))
        n_pairs = (n_selected * (n_selected - 1)).clamp(min=1)
        redundancy_conflict = pair_conflict / n_pairs

        return (self.w_acc * accuracy + self.w_cov * coverage
                - self.w_conflict * redundancy_conflict)

    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        raw = self.score(s)                       # có thể âm
        return torch.exp(self.beta * raw)          # đảm bảo R > 0