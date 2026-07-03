import pytest
import torch

from src.models.cnn import CNNBaseline


@pytest.mark.parametrize("stage", ["head_only", "last_block", "full"])
def test_freeze_stage_valid(stage):
    model = CNNBaseline(num_classes=3, freeze_stage=stage)
    head_trainable = any(p.requires_grad for p in model.backbone.classifier.parameters())
    assert head_trainable, "Classifier head phải luôn trainable ở mọi stage"


def test_head_only_freezes_backbone():
    model = CNNBaseline(num_classes=3, freeze_stage="head_only")
    assert all(not p.requires_grad for p in model.backbone.features.parameters())


def test_full_unfreezes_everything():
    model = CNNBaseline(num_classes=3, freeze_stage="full")
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_invalid_stage_raises():
    with pytest.raises(ValueError):
        CNNBaseline(num_classes=3, freeze_stage="not_a_stage")


def test_forward_shapes():
    model = CNNBaseline(num_classes=5, freeze_stage="head_only")
    x = torch.randn(2, 3, 224, 224)
    logits, features = model(x)
    assert logits.shape == (2, 5)
    assert features.shape == (2, 1280)
