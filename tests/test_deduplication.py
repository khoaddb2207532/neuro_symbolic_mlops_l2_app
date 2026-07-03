import torch

from src.models.cnn import CNNBaseline, FeatureExtractor
from src.rules.io import save_rules_excel
from src.rules.penalty import BinaryTransformer, VectorizedRulePenalty
from src.rules.rule_types import Condition, Rule, RuleSet


def test_no_duplicate_cnn_alias():
    """CNNWithFeatures đã bị xoá — chỉ còn CNNBaseline duy nhất."""
    import src.models.cnn as cnn_module

    assert not hasattr(cnn_module, "CNNWithFeatures")


def test_single_rule_penalty_implementation():
    """compute_rule_penalty (loop, không label smoothing) đã bị xoá — chỉ còn
    VectorizedRulePenalty (có label smoothing) để tránh 2 công thức lệch nhau."""
    import src.rules.penalty as penalty_module

    assert not hasattr(penalty_module, "compute_rule_penalty")
    assert not hasattr(penalty_module, "DynamicBinaryTransformer")


def test_vectorized_penalty_has_no_dead_temperature_state():
    rule = Rule([Condition(0, "<=", 0.5)], target_class=0)
    penalty = VectorizedRulePenalty(RuleSet(rules=[rule]), penalty_weight=0.1, num_classes=3)
    assert not hasattr(penalty, "update_temperature")
    assert not hasattr(penalty, "_temperature")


def test_vectorized_penalty_forward_runs():
    rule = Rule([Condition(0, "<=", 0.5)], target_class=1)
    penalty = VectorizedRulePenalty(RuleSet(rules=[rule]), penalty_weight=0.1, num_classes=3)
    features = torch.rand(4, 4)
    logits = torch.randn(4, 3)
    loss = penalty(features, logits)
    assert loss.dim() == 0  # scalar


def test_save_rules_excel_shared_across_pipeline(tmp_path):
    rules = [Rule([Condition(0, "<=", 0.5)], target_class=0, confidence=0.9)]
    out_path = tmp_path / "rules.xlsx"
    save_rules_excel(rules, str(out_path))
    assert out_path.exists()


def test_feature_extractor_shares_backbone_factory():
    """CNNBaseline và FeatureExtractor phải cho ra kiến trúc backbone giống hệt
    nhau (cùng dùng _build_mobilenet_v3), tránh code xây dựng model bị lặp."""
    model_a = CNNBaseline(num_classes=5, freeze_stage="head_only")
    model_b = FeatureExtractor(num_classes=5)
    keys_a = set(model_a.backbone.state_dict().keys())
    keys_b = set(model_b.backbone.state_dict().keys())
    assert keys_a == keys_b
