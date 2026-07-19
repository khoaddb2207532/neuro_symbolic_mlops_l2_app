import pytest

from src.data.protocol import validate_audit_summary, validate_split_protocol


def test_validation_split_must_be_large_enough():
    with pytest.raises(ValueError, match="too small"):
        validate_split_protocol({"train": 90, "val": 5, "test": 5}, [0, 0, 1, 1, 1],
                                min_val_fraction=0.15, min_val_samples_per_class=2)


def test_validation_requires_enough_samples_in_every_present_class():
    with pytest.raises(ValueError, match="too few"):
        validate_split_protocol({"train": 60, "val": 20, "test": 20}, [0] * 19 + [1],
                                min_val_fraction=0.15, min_val_samples_per_class=2)


def test_valid_protocol_returns_auditable_summary():
    result = validate_split_protocol({"train": 60, "val": 20, "test": 20},
                                     [0] * 10 + [1] * 10,
                                     min_val_fraction=0.15, min_val_samples_per_class=10)
    assert result["validation_fraction"] == pytest.approx(0.2)
    assert result["validation_class_counts"] == {"0": 10, "1": 10}


def test_cross_split_near_duplicates_block_baseline():
    with pytest.raises(ValueError, match="near-duplicate"):
        validate_audit_summary({"duplicates": {"n_near_cross_split_pairs": 3}})
