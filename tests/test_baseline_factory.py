import pytest

from src.models.cnn import (
    canonical_baseline_name,
    selected_baseline_checkpoint,
    selected_baseline_metrics,
)


@pytest.mark.parametrize("alias, expected", [
    ("efficientnet", "efficientnet_b0"),
    ("effcientnet", "efficientnet_b0"),
    ("swinT", "swin_t"),
    ("ViT", "vit_b_16"),
    ("resnet50", "resnet50"),
])
def test_canonical_baseline_aliases(alias, expected):
    assert canonical_baseline_name(alias) == expected


def test_unknown_baseline_is_rejected():
    with pytest.raises(ValueError, match="Unsupported baseline"):
        canonical_baseline_name("unknown_net")


def test_selected_baseline_artifact_paths():
    params = {"output_dir": "out", "feature_extraction": {"architecture": "vit"}}
    assert selected_baseline_checkpoint(params).replace("\\", "/") == \
        "out/01_baselines/vit_b_16/model.pt"
    assert selected_baseline_metrics(params).replace("\\", "/") == \
        "out/01_baselines/vit_b_16/metrics.json"


def test_explicit_checkpoint_takes_precedence():
    params = {"feature_extraction": {
        "architecture": "swinT", "checkpoint_path": "custom/model.pt",
    }}
    assert selected_baseline_checkpoint(params) == "custom/model.pt"
