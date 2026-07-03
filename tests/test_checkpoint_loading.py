import torch
import torch.nn as nn

from src.models.cnn import CNNBaseline
from src.training.trainer import train_one_epoch
from src.utils.checkpoint import load_model_weights


def test_load_model_weights_missing_file_not_required(tmp_path):
    model = CNNBaseline(num_classes=3, freeze_stage="head_only")
    ok = load_model_weights(model, str(tmp_path / "missing.pth"), device="cpu", required=False)
    assert ok is False


def test_load_model_weights_missing_file_required_raises(tmp_path):
    import pytest

    model = CNNBaseline(num_classes=3, freeze_stage="head_only")
    with pytest.raises(FileNotFoundError):
        load_model_weights(model, str(tmp_path / "missing.pth"), device="cpu", required=True)


def test_load_model_weights_handles_full_checkpoint_format(tmp_path):
    """Mô phỏng đúng định dạng do save_checkpoint() lưu (dict có
    'model_state_dict') — trước đây FeatureExtractor load sai định dạng này."""
    model_a = CNNBaseline(num_classes=3, freeze_stage="full")
    ckpt_path = tmp_path / "baseline_best.pth"
    torch.save(
        {
            "epoch": 5,
            "model_state_dict": model_a.state_dict(),
            "optimizer_state_dict": {},
            "best_acc": 0.9,
            "class_names": ["a", "b", "c"],
        },
        ckpt_path,
    )

    model_b = CNNBaseline(num_classes=3, freeze_stage="full")
    ok = load_model_weights(model_b, str(ckpt_path), device="cpu", required=True)
    assert ok is True
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b)


def test_load_model_weights_handles_raw_state_dict_format(tmp_path):
    """Mô phỏng định dạng state_dict thô (vd final_model_weights.pth)."""
    model_a = CNNBaseline(num_classes=3, freeze_stage="full")
    ckpt_path = tmp_path / "final_model_weights.pth"
    torch.save(model_a.state_dict(), ckpt_path)

    model_b = CNNBaseline(num_classes=3, freeze_stage="full")
    ok = load_model_weights(model_b, str(ckpt_path), device="cpu", required=True)
    assert ok is True


def test_train_one_epoch_returns_train_acc():
    model = CNNBaseline(num_classes=3, freeze_stage="head_only")
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 3, (4,))
    loader = [(images, labels)]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    result = train_one_epoch(model, loader, criterion, optimizer, device="cpu")
    assert len(result) == 4
    train_loss, train_ce, train_penalty, train_acc = result
    assert 0.0 <= train_acc <= 1.0
