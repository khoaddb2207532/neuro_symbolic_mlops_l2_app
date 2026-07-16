from typing import Optional

import torch


class RuleSetReward:
    def __init__(self, cover: torch.Tensor, correct: torch.Tensor,
                 rule_len: torch.Tensor, max_rules: int,
                 alpha: float = 1.0, lambda_1: float = 1.0, lambda_2: float = 1.0,
                 lambda_3: float = 1.0, K: int = 10, gamma: float = 3.0,
                 sample_weight: Optional[torch.Tensor] = None):
        """
        Bản torch, có batch, của công thức R(S) mô tả trong bản tham chiếu
        numpy `calculate_reward`. Khác với bản `calculate_reward` (nhận
        `candidate_pool` là dict + `S` là list[int] rời rạc, xử lý TỪNG tập
        luật một), lớp này giữ nguyên cách biểu diễn cũ của file — mọi luật
        được mã hoá sẵn thành ma trận (n_rules, n_val), và một tập luật S
        được biểu diễn bằng vector nhị phân s (n_rules,) — để có thể tính
        reward cho CẢ MỘT BATCH tập luật (B, n_rules) cùng lúc trên GPU,
        thay vì lặp for như bản numpy.

        cover:    (n_rules, n_val) bool - luật i có match mẫu j không.
        correct:  (n_rules, n_val) bool - luật i match VÀ dự đoán đúng nhãn
                  mẫu j. Dùng để suy ra `err` (tỉ lệ sai số) của từng luật,
                  tương đương trường `err` trong `candidate_pool[r]`.
        rule_len: (n_rules,) - số điều kiện (mệnh đề AND) trong luật, tương
                  đương trường `len` trong `candidate_pool[r]`.
        max_rules: giữ lại như cũ (không dùng trực tiếp trong công thức, chỉ
                  để tham chiếu/ràng buộc bên ngoài nếu cần).

        alpha, lambda_1, lambda_2, lambda_3, K, gamma: đúng như mô tả trong
        `params` của bản numpy:
          - alpha:    phạt độ dài của TỪNG luật đơn lẻ (trong f_quality).
          - lambda_1: trọng số cho f_cover (khuyến khích tổng độ phủ).
          - lambda_2: trọng số phạt f_overlap (trùng lặp biên quyết định).
          - lambda_3: trọng số phạt f_size (kích thước tập luật).
          - K:        số lượng luật tối đa kỳ vọng (dùng trong f_size).
          - gamma:    nhiệt độ nghịch đảo, kiểm soát mode-matching khi
                      exponentiate U(S) -> R(S).

        sample_weight: (n_val,) float, optional — trọng số uncertainty/độ
                  khó của TỪNG mẫu validation (xem `uncertainty.py`,
                  `compute_sample_weight*`). KHÔNG cần chuẩn hoá tổng trước
                  khi truyền vào — `f_cover` tự chia cho tổng trọng số bên
                  dưới. Nếu None (mặc định), tương đương mọi mẫu có trọng số
                  1.0, tức `f_cover` quay lại đúng tỉ lệ union-coverage
                  KHÔNG trọng số như trước (tương thích ngược hoàn toàn).
                  LƯU Ý THỨ TỰ HÀNG: `sample_weight[j]` phải cùng permutation
                  mẫu với cột j của `cover`/`correct` — nếu `cover`/`correct`
                  bị hoán vị (vd permute theo luật ở pipeline.py) thì đó là
                  permutation theo LUẬT (chiều 0, n_rules), không ảnh hưởng
                  chiều mẫu (chiều 1, n_val) nên `sample_weight` không cần
                  hoán vị lại theo permutation đó.

        Chỉ `f_cover` được tính theo trọng số (đúng mục đích nêu trong
        `uncertainty.py`: "chuyển coverage ... thành tỉ lệ TRỌNG SỐ LỖI/
        UNCERTAINTY được phủ"). `f_quality` (freq_r/err_r/q_r của từng luật)
        và `f_overlap` (đo trùng lặp biên quyết định, không phải độ khó của
        mẫu) vẫn tính KHÔNG trọng số như thiết kế gốc của bản này.

        LƯU Ý THAY ĐỔI SO VỚI BẢN TRƯỚC CỦA FILE NÀY: bản trước tách riêng
        `accuracy` (đo trên phần được phủ CHUNG của cả tập luật) và
        `conflict_ratio` (chỉ tính xung đột GIỮA CÁC TARGET khác nhau, coi
        2 luật cùng target phủ cùng vùng là vô hại). Theo mô tả mới, `err`
        được tính RIÊNG CHO TỪNG LUẬT (không phụ thuộc luật nào khác được
        chọn cùng) và gộp vào `f_quality` per-rule, còn `f_overlap` đếm MỌI
        mẫu bị phủ bởi > 1 luật trong S — không phân biệt luật đó có cùng
        target hay không. Đây là lựa chọn thiết kế của mô tả mới (giống bản
        `calculate_reward` numpy), không phải lỗi — nếu vẫn muốn phân biệt
        theo target, cần truyền lại `targets` và khôi phục logic
        `class_masks` như bản trước.
        """
        self.cover = cover.float()
        self.correct = correct.float()
        self.rule_len = rule_len.float()
        self.n_val = cover.shape[1]
        self.n_rules = cover.shape[0]
        self.max_rules = max_rules

        self.alpha = alpha
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.K = K
        self.gamma = gamma

        # sample_weight (n_val,): trọng số uncertainty theo mẫu, dùng để
        # weighted-coverage trong f_cover. Mặc định None -> toàn 1.0, khớp
        # hành vi union-coverage không trọng số của bản trước.
        if sample_weight is None:
            self.sample_weight = torch.ones(self.n_val, device=self.cover.device)
        else:
            self.sample_weight = sample_weight.to(self.cover.device).float()
            assert self.sample_weight.shape == (self.n_val,), (
                "sample_weight phải có shape (n_val,) khớp số cột của cover/correct"
            )
        self._weight_sum = self.sample_weight.sum().clamp(min=1e-8)

        # --- Thống kê TĨNH theo từng luật (không đổi theo S được chọn) ---
        # freq_r: tần suất phủ của luật r (tương đương candidate_pool[r]['freq'])
        self.freq = self.cover.sum(-1) / self.n_val                        # (n_rules,)

        # err_r: tỉ lệ sai số của luật r TRÊN PHẦN NÓ PHỦ, tức
        # err_r = 1 - accuracy_r (tương đương candidate_pool[r]['err']).
        cover_count = self.cover.sum(-1).clamp(min=1)
        self.err = 1.0 - (self.correct.sum(-1) / cover_count)              # (n_rules,)

        # q_r = freq_r * (1 - err_r) * exp(-alpha * len_r): điểm chất lượng
        # nội tại của luật r, giống hệt công thức q_r trong bản numpy.
        self.q = self.freq * (1.0 - self.err) * torch.exp(-self.alpha * self.rule_len)  # (n_rules,)

    def components(self, s: torch.Tensor) -> dict:
        """Tách riêng từng thành phần của U(S), dùng chung bởi `score()`.

        s: (B, n_rules) nhị phân (0/1) — mỗi hàng là một tập luật S.
        Trả về dict các tensor (B,):
          - n_selected: |S|
          - f_quality:  sum_{r in S} q_r
          - f_cover:    tỉ lệ TRỌNG SỐ (sample_weight) của các mẫu được phủ
                        bởi >=1 luật trong S, chia cho tổng trọng số mọi
                        mẫu. Khi sample_weight=None (mặc định toàn 1.0),
                        đây chính là |union cov(r), r in S| / n_val như cũ.
          - f_overlap:  tỉ lệ mẫu (không trọng số) bị phủ bởi > 1 luật trong S
          - f_size:     max(0, |S| - K)
        """
        n_selected = s.sum(-1)                                      # (B,)

        # f_quality(S) = sum q_r trên các luật được chọn = s @ q
        f_quality = s @ self.q                                       # (B,)

        # counts[b, j] = số luật (trong tập S của batch b) phủ mẫu j
        counts = s @ self.cover                                      # (B, n_val)

        # f_cover: tỉ lệ TRỌNG SỐ của mẫu được phủ bởi ÍT NHẤT 1 luật
        # (weighted union coverage) — nhân mask covered với sample_weight
        # trước khi chia cho tổng trọng số, thay vì chia đều cho n_val.
        covered = (counts > 0).float()                                # (B, n_val)
        f_cover = (covered * self.sample_weight).sum(-1) / self._weight_sum  # (B,)

        # f_overlap: tỉ lệ mẫu bị phủ bởi NHIỀU HƠN 1 luật (trùng biên quyết định)
        f_overlap = (counts > 1).float().sum(-1) / self.n_val         # (B,)

        # f_size: phạt tuyến tính nếu số luật chọn vượt quá K kỳ vọng
        f_size = (n_selected - self.K).clamp(min=0)                   # (B,)

        return {
            "n_selected": n_selected,
            "f_quality": f_quality,
            "f_cover": f_cover,
            "f_overlap": f_overlap,
            "f_size": f_size,
        }

    def score(self, s: torch.Tensor) -> torch.Tensor:
        """U(S) = f_quality + lambda_1*f_cover - lambda_2*f_overlap - lambda_3*f_size"""
        c = self.components(s)
        return (c["f_quality"]
                + self.lambda_1 * c["f_cover"]
                - self.lambda_2 * c["f_overlap"]
                - self.lambda_3 * c["f_size"])

    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        """R(S) = exp(gamma * U(S)) -- đảm bảo phần thưởng dương.

        LƯU Ý về tập rỗng: khi |S| = 0, mọi thành phần trong `components()`
        đều bằng 0 -> U(S) = 0 -> exp(gamma*0) = 1. Bản numpy tham chiếu
        trả về hẳn 0.0 cho tập rỗng (return 0.0 sớm trước khi tính U(S)).
        Nếu muốn giữ đúng hành vi đó (thay vì reward = 1 cho tập rỗng),
        dùng đoạn ép về 0 bên dưới.
        """
        raw = self.score(s)
        reward = torch.exp(self.gamma * raw)
        # Ép reward = 0 cho các hàng s rỗng (|S| = 0), khớp hành vi
        # "return 0.0" của bản numpy tham chiếu:
        reward = torch.where(s.sum(-1) > 0, reward, torch.zeros_like(reward))
        return reward