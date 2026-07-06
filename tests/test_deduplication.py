import pytest
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


def test_vectorized_penalty_temperature_is_real_and_used():
    """Trước đây update_temperature() là no-op (dead code) nên đã bị xoá.
    Nay đã nâng cấp lên khớp mềm CÓ dùng nhiệt độ thật trong forward() — test
    này khoá lại việc nhiệt độ thực sự ảnh hưởng tới rule_sat/loss, tránh
    quay lại tình trạng "gọi update_temperature nhưng không có tác dụng"."""
    rule = Rule([Condition(0, "<=", 0.5)], target_class=0)
    penalty = VectorizedRulePenalty(
        RuleSet(rules=[rule]), penalty_weight=0.1, num_classes=3,
        initial_temp=1.0, final_temp=50.0,
    )
    assert hasattr(penalty, "update_temperature")
    assert hasattr(penalty, "_temperature")
    assert penalty._temperature.item() == pytest.approx(1.0)

    features = torch.tensor([[0.5]])  # đúng ngay tại ngưỡng -> sigmoid(0)=0.5 dù nhiệt độ nào
    logits = torch.zeros(1, 3)

    penalty.update_temperature(epoch=0, total_epochs=10)
    temp_start = penalty._temperature.item()
    loss_soft = penalty(features, logits)

    penalty.update_temperature(epoch=9, total_epochs=10)
    temp_end = penalty._temperature.item()
    loss_sharp = penalty(features, logits)

    assert temp_start < temp_end  # nhiệt độ phải tăng dần (mềm -> cứng)
    # Với feature nằm lệch khỏi ngưỡng, rule_sat (và do đó loss) phải khác
    # nhau rõ rệt giữa nhiệt độ thấp (mềm) và nhiệt độ cao (cứng).
    features_offset = torch.tensor([[0.3]])
    penalty.update_temperature(epoch=0, total_epochs=10)
    sat_soft = penalty(features_offset, logits)
    penalty.update_temperature(epoch=9, total_epochs=10)
    sat_sharp = penalty(features_offset, logits)
    assert sat_soft.item() != pytest.approx(sat_sharp.item())


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
