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
    batch_norm_params = {
        id(p)
        for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
        for p in module.parameters()
    }
    assert all(
        p.requires_grad for p in model.backbone.parameters()
        if id(p) not in batch_norm_params
    )
    assert all(
        not p.requires_grad for p in model.backbone.parameters()
        if id(p) in batch_norm_params
    )


def test_batchnorm_stays_frozen_in_train_mode():
    model = CNNBaseline(num_classes=3, freeze_stage="full")
    model.train()
    model.freeze_bn()
    batch_norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert batch_norms
    assert all(not m.training for m in batch_norms)
    assert all(not p.requires_grad for m in batch_norms for p in m.parameters())


def test_invalid_stage_raises():
    with pytest.raises(ValueError):
        CNNBaseline(num_classes=3, freeze_stage="not_a_stage")


def test_forward_shapes():
    model = CNNBaseline(num_classes=5, freeze_stage="head_only")
    x = torch.randn(2, 3, 224, 224)
    logits, features = model(x)
    assert logits.shape == (2, 5)
    assert features.shape == (2, 1280)
