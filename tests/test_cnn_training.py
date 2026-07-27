import torch

from src.models.cnn import (
    BASELINE_ARCHITECTURES,
    CNNBaseline,
    ImageClassificationBaseline,
)
from src.training.optimizer import build_optimizer


def test_cnn_finetunes_all_parameters():
    model = CNNBaseline(num_classes=3)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_fixed_differential_lr_groups():
    model = CNNBaseline(num_classes=5)
    optimizer = build_optimizer(
        model,
        {"lr_head": 1e-3, "lr_backbone": 1e-4, "weight_decay": 1e-4},
    )

    assert len(optimizer.param_groups) == 2
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 1e-4]
    assert [group["name"] for group in optimizer.param_groups] == ["head", "backbone"]


def test_forward_shapes():
    model = CNNBaseline(num_classes=5)
    logits, features = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 5)
    assert features.shape == (2, 1280)


def test_comparison_baseline_architectures_are_constructible():
    for architecture in BASELINE_ARCHITECTURES:
        model = ImageClassificationBaseline(
            architecture=architecture,
            num_classes=5,
            pretrained=False,
        )
        assert all(parameter.requires_grad for parameter in model.parameters())
