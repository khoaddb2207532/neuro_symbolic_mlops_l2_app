import pickle

import torch
from sklearn.ensemble import RandomForestClassifier

from src.rules.extractor import RuleExtractor
from src.rules.validator import GPUFastRuleValidator


def test_validate_crossval_removed():
    """validate_crossval từng tự train lại K RandomForest trên các fold của
    train — trùng lặp với train_and_save_rf(). Đã bỏ hoàn toàn."""
    assert not hasattr(GPUFastRuleValidator, "validate_crossval")
    assert not hasattr(GPUFastRuleValidator, "_deduplicate")


def test_train_and_save_rf_extracts_rules_in_one_place(tmp_path):
    """train_and_save_rf() giờ là nơi DUY NHẤT train RF + trích luật thô —
    trả về RuleSet luôn, không cần bước RuleExtractor riêng ở nơi gọi."""
    import numpy as np

    from src.data.features import train_and_save_rf

    # Chuẩn bị dữ liệu train giả lập trên đĩa đúng định dạng mà hàm mong đợi
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    X = torch.randn(50, 8)
    y = torch.randint(0, 3, (50,))
    torch.save(X, features_dir / "train_features.pt")
    torch.save(y, features_dir / "train_labels.pt")

    raw_rules = train_and_save_rf(
        features_dir=str(features_dir),
        rf_output_path=str(tmp_path / "rf.joblib"),
        rules_output_path=str(tmp_path / "raw_rules.pkl"),
        n_estimators=5,
    )

    assert len(raw_rules) > 0
    assert (tmp_path / "rf.joblib").exists()
    assert (tmp_path / "raw_rules.pkl").exists()

    with open(tmp_path / "raw_rules.pkl", "rb") as f:
        pickled_rules = pickle.load(f)
    assert len(pickled_rules) == len(raw_rules)


def test_validate_filters_by_val_set_no_retraining(tmp_path):
    """Lọc luật (validate) không được tự train RF nào cả — chỉ nhận rule_set
    có sẵn + val features/labels."""
    X_train = torch.rand(40, 5).numpy()
    y_train = torch.randint(0, 2, (40,)).numpy()
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    rf.fit(X_train, y_train)
    raw_rules = RuleExtractor().extract(rf)

    val_features = torch.rand(20, 5)
    val_labels = torch.randint(0, 2, (20,))

    validator = GPUFastRuleValidator(min_supp=0.0, min_conf=0.0)
    valid_rules = validator.validate(raw_rules, val_features, val_labels)
    assert len(valid_rules) <= len(raw_rules)
