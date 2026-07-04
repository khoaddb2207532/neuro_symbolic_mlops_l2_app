import torch


class RuleSetReward:
    def __init__(self, cover: torch.Tensor, correct: torch.Tensor,
                 rule_len: torch.Tensor, max_rules: int,
                 w_acc=1.0, w_cov=0.5, w_red=0.3, w_comp=0.2, beta=3.0):
        """
        cover:   (n_rules, n_val) bool - luật i có match mẫu j không
        correct: (n_rules, n_val) bool - luật i match VÀ dự đoán đúng nhãn mẫu j
        rule_len: (n_rules,) - số điều kiện trong luật (đo độ phức tạp)
        """
        self.cover = cover.float()
        self.correct = correct.float()
        self.rule_len = rule_len.float()
        self.n_val = cover.shape[1]
        self.max_rules = max_rules
        self.w_acc, self.w_cov, self.w_red, self.w_comp = w_acc, w_cov, w_red, w_comp
        self.beta = beta
        # ma trận overlap giữa các luật (Jaccard) để phạt trùng lặp — tính 1 lần
        inter = cover.float() @ cover.float().T
        card = cover.float().sum(-1, keepdim=True)
        union = card + card.T - inter
        self.jaccard = inter / union.clamp(min=1e-8)   # (n_rules, n_rules)

    def score(self, s: torch.Tensor) -> torch.Tensor:
        # s: (B, n_rules) nhị phân
        n_selected = s.sum(-1).clamp(min=1)

        covered = (s @ self.cover) > 0                       # (B, n_val)
        correct_covered = (s @ self.correct) > 0
        coverage = covered.float().mean(-1)
        # accuracy chỉ tính trên phần được phủ
        accuracy = (correct_covered.float().sum(-1)) / covered.float().sum(-1).clamp(min=1)

        # redundancy trung bình giữa các cặp luật đã chọn
        pair_red = (s.unsqueeze(1) * s.unsqueeze(2) * self.jaccard).sum((-1, -2))
        n_pairs = (n_selected * (n_selected - 1)).clamp(min=1)
        redundancy = pair_red / n_pairs

        complexity = n_selected / self.max_rules

        return (self.w_acc * accuracy + self.w_cov * coverage
                - self.w_red * redundancy - self.w_comp * complexity)

    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        raw = self.score(s)                       # có thể âm
        return torch.exp(self.beta * raw)          # đảm bảo R > 0