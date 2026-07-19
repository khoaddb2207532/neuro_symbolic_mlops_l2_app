from pathlib import Path

from PIL import Image

from src.data.audit import audit_dataset, protocol_violations, write_audit_report


def _image(path: Path, color, size=(16, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def test_audit_detects_cross_split_exact_duplicate_and_writes_reports(tmp_path):
    data = tmp_path / "data"
    _image(data / "train" / "a" / "object1_train.png", (255, 0, 0))
    _image(data / "train" / "b" / "object2_train.png", (0, 255, 0))
    _image(data / "val" / "a" / "object1_val.png", (255, 0, 0))
    _image(data / "val" / "b" / "object3_val.png", (0, 0, 255))
    _image(data / "test" / "a" / "object4_test.png", (125, 125, 125))
    _image(data / "test" / "b" / "object5_test.png", (30, 30, 30))

    records, summary, duplicates = audit_dataset(data, group_regex=r"^(object\d+)")

    assert len(records) == 6
    assert summary["n_corrupt_images"] == 0
    assert summary["duplicates"]["n_exact_cross_split_groups"] == 1
    assert summary["groups"]["n_cross_split_groups"] == 1
    assert protocol_violations(summary, fail_on_leakage=True)

    output = tmp_path / "report"
    write_audit_report(output, records, summary, duplicates)
    assert (output / "summary.json").is_file()
    assert (output / "image_manifest.csv").is_file()
    assert (output / "class_distribution.csv").is_file()
    assert (output / "duplicates.csv").is_file()


def test_audit_detects_corrupt_image_and_class_mismatch(tmp_path):
    data = tmp_path / "data"
    for split in ("train", "val", "test"):
        _image(data / split / "a" / f"{split}.png", (1, 2, 3))
    corrupt = data / "train" / "a" / "broken.jpg"
    corrupt.write_bytes(b"not an image")
    _image(data / "train" / "b" / "only_train.png", (4, 5, 6))

    _, summary, _ = audit_dataset(data)

    assert summary["n_corrupt_images"] == 1
    assert summary["class_mismatch"] == {"val": ["b"], "test": ["b"]}
    violations = protocol_violations(
        summary, fail_on_corrupt=True, fail_on_class_mismatch=True
    )
    assert len(violations) == 2
