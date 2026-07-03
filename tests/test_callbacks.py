import pytest

from src.training.callbacks import EarlyStopping


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        EarlyStopping(patience=2, mode="not_a_mode")


def test_first_call_is_always_best_max_mode():
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper(0.5) is True
    assert stopper.early_stop is False


def test_max_mode_higher_is_better():
    """mode='max' (vd val_acc): giá trị CAO hơn mới là cải thiện."""
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper(0.5) is True     # baseline
    assert stopper(0.4) is False    # thấp hơn -> KHÔNG phải cải thiện, counter=1
    assert stopper.early_stop is False
    assert stopper(0.6) is True     # cao hơn -> cải thiện, counter reset


def test_min_mode_lower_is_better():
    """mode='min' (vd val_loss): giá trị THẤP hơn mới là cải thiện — đây chính
    là hành vi trước đây bị SAI khi dùng chung logic 'cao hơn là tốt' cho cả
    val_acc lẫn val_loss."""
    stopper = EarlyStopping(patience=2, mode="min")
    assert stopper(1.0) is True     # baseline
    assert stopper(1.2) is False    # cao hơn -> tệ hơn với loss, counter=1
    assert stopper.early_stop is False
    assert stopper(0.8) is True     # thấp hơn -> cải thiện, counter reset
    assert stopper.counter == 0


def test_min_mode_triggers_early_stop_on_no_improvement():
    stopper = EarlyStopping(patience=2, mode="min")
    stopper(1.0)   # baseline
    stopper(1.1)   # counter=1
    assert stopper.early_stop is False
    stopper(1.2)   # counter=2 -> patience reached
    assert stopper.early_stop is True


def test_no_save_side_effects():
    """EarlyStopping không được có thuộc tính/hành vi ghi file — việc lưu
    checkpoint thuộc về trainer.py, tránh trùng lặp trách nhiệm."""
    stopper = EarlyStopping(patience=2)
    assert not hasattr(stopper, "path")
    assert not hasattr(stopper, "_save")
