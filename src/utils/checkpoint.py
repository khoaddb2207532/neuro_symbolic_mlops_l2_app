"""Tiện ích nạp trọng số model từ checkpoint — dùng chung cho mọi nơi cần load
lại một model đã huấn luyện (stage2 feature extraction, stage5 rule-regularized
fine-tuning, app serving).

Repo hiện có 2 định dạng checkpoint khác nhau, và một nơi đang nạp sai định
dạng một cách âm thầm trước khi hàm này tồn tại:
  - `save_checkpoint()` (trainer.py) lưu dict đầy đủ:
        {"epoch", "model_state_dict", "optimizer_state_dict", "best_acc", "class_names"}
    → dùng cho *_best.pth (baseline_best.pth, rule_regularized_best.pth)
  - `torch.save(model.state_dict(), ...)` lưu state_dict thô
    → dùng cho final_model_weights.pth

`load_model_weights()` tự phát hiện định dạng nào đang gặp, tránh phải nhớ
"file này phải load kiểu gì" ở từng nơi gọi.
"""
import os

import torch
import torch.nn as nn

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_model_weights(model: nn.Module, path: str, device, required: bool = False) -> bool:
    """Nạp trọng số vào `model` từ `path`, tự nhận diện định dạng checkpoint.

    Trả về True nếu nạp thành công.
    Trả về False nếu không tìm thấy file và `required=False` (model giữ
    nguyên trọng số hiện có, vd pretrained ImageNet — dùng cho trường hợp
    checkpoint chưa tồn tại, ví dụ chạy stage5 trước khi có stage1).
    Raise FileNotFoundError nếu `required=True` và không tìm thấy file.
    """
    if not os.path.exists(path):
        msg = f"Không tìm thấy checkpoint tại {path}."
        if required:
            raise FileNotFoundError(
                f"{msg} Bắt buộc phải có checkpoint này (required=True) — "
                "hãy chạy stage trước đó (vd `dvc repro train_baseline`) trước."
            )
        logger.warning("%s Bỏ qua — model giữ nguyên trọng số khởi tạo hiện tại.", msg)
        return False

    obj = torch.load(path, map_location=device)
    state_dict = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model.load_state_dict(state_dict)
    logger.info("Đã nạp trọng số model từ %s", path)
    return True
