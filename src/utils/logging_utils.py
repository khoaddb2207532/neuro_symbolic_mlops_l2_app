"""Structured logging cấu hình tập trung cho toàn bộ pipeline.

MLOps L2 yêu cầu log có thể theo dõi, tổng hợp và tra cứu được (không chỉ print()
rải rác trong notebook). Mọi module nên gọi get_logger(__name__) thay vì print().
"""
import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # tránh add handler trùng khi gọi lại nhiều lần
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(Path(log_dir) / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
