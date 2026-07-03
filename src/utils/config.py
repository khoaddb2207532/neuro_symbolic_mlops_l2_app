"""Nạp cấu hình tập trung từ params.yaml (single source of truth).

Thay vì hard-code hằng số rải rác trong notebook (NUM_EPOCHS, LR, DATA_DIR, ...),
toàn bộ tham số nằm trong params.yaml ở gốc repo. Điều này là điều kiện bắt buộc
của MLOps cấp độ 2: pipeline phải "tái lập" (reproducible) chỉ bằng cách track
1 file config, và DVC có thể tự phát hiện khi tham số đổi để trigger lại đúng
stage bị ảnh hưởng.
"""
from pathlib import Path
from typing import Any, Dict

import yaml


def load_params(config_path: str = "params.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file cấu hình: {path.resolve()}. "
            "Hãy chắc chắn bạn chạy lệnh từ thư mục gốc của project."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
