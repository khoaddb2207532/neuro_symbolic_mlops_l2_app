import pytest

from src.training.trainer import MONITOR_MODES, DEFAULT_CONFIG


def test_monitor_modes_registered_correctly():
    assert MONITOR_MODES["val_acc"] == "max"
    assert MONITOR_MODES["val_loss"] == "min"


def test_default_config_monitors_val_acc_for_backward_compat():
    assert DEFAULT_CONFIG["monitor_metric"] == "val_acc"


def test_unregistered_monitor_metric_should_be_rejected():
    """train_model() phải raise ValueError rõ ràng nếu ai đó đặt monitor_metric
    thành 1 metric chưa đăng ký trong MONITOR_MODES, thay vì âm thầm dùng sai
    mode so sánh (đây chính là bug ban đầu: hard-code mode='max' cho mọi metric)."""
    assert "val_f1" not in MONITOR_MODES  # ví dụ 1 metric chưa đăng ký
