"""Callback dừng sớm khi metric theo dõi không cải thiện.

Hỗ trợ `mode="max"` (metric càng cao càng tốt, vd val_acc, F1) hoặc
`mode="min"` (càng thấp càng tốt, vd val_loss) 
"""


class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0, mode: str = "max"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode phải là 'min' hoặc 'max', nhận '{mode}'")
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def _is_improvement(self, value: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return value > self.best_score + self.delta
        return value < self.best_score - self.delta

    def __call__(self, value: float) -> bool:
        """
        Trả về True nếu đây là điểm tốt nhất từ trước tới nay (tín hiệu để
        trainer quyết định có lưu checkpoint hay không), False nếu không.
        """
        is_best = self._is_improvement(value)
        if is_best:
            self.best_score = value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience} (mode={self.mode})")
            if self.counter >= self.patience:
                self.early_stop = True
        return is_best
