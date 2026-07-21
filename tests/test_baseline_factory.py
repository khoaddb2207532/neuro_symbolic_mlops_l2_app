import pytest

from src.models.cnn import canonical_baseline_name


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
