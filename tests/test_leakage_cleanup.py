import csv
from pathlib import Path

from PIL import Image

from src.data.leakage_cleanup import apply_deletion_plan, build_deletion_plan


FIELDS = (
    "duplicate_type", "group_id", "path", "split", "class_name", "sha256",
    "path_a", "split_a", "class_a", "path_b", "split_b", "class_b",
    "hamming_distance", "cross_split", "cross_class",
)


def _image(path: Path, color, size=(12, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _write_duplicates(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_plan_keeps_higher_resolution_for_exact_and_distance_one(tmp_path):
    data = tmp_path / "data"
    train_exact = data / "train" / "a" / "train_exact.png"
    val_exact = data / "val" / "a" / "val_exact.png"
    train_near = data / "train" / "a" / "train_near.png"
    test_near = data / "test" / "a" / "test_near.png"
    for index in range(5):
        _image(data / "train" / "a" / f"extra_{index}.png", (index, index, index))
    # Exact duplicate co cung dimension, nen tie-break giu val (split nho).
    _image(train_exact, (255, 0, 0), size=(12, 12))
    _image(val_exact, (255, 0, 0), size=(12, 12))
    # Anh train co do phan giai cao hon test, nen giu train du train lon hon.
    _image(train_near, (0, 255, 0), size=(30, 20))
    _image(test_near, (0, 250, 0), size=(10, 10))

    report = tmp_path / "duplicates.csv"
    _write_duplicates(report, [
        {"duplicate_type": "exact", "group_id": "1", "path": "train/a/train_exact.png",
         "split": "train", "class_name": "a", "cross_split": "True", "cross_class": "False"},
        {"duplicate_type": "exact", "group_id": "1", "path": "val/a/val_exact.png",
         "split": "val", "class_name": "a", "cross_split": "True", "cross_class": "False"},
        {"duplicate_type": "near", "path_a": "train/a/train_near.png", "split_a": "train",
         "class_a": "a", "path_b": "test/a/test_near.png", "split_b": "test", "class_b": "a",
         "hamming_distance": "1", "cross_split": "True", "cross_class": "False"},
        {"duplicate_type": "near", "path_a": "train/a/train_exact.png", "split_a": "train",
         "class_a": "a", "path_b": "test/a/test_near.png", "split_b": "test", "class_b": "a",
         "hamming_distance": "2", "cross_split": "True", "cross_class": "False"},
    ])

    plan, summary = build_deletion_plan(data, report)

    assert {item.path for item in plan} == {
        "train/a/train_exact.png", "test/a/test_near.png"
    }
    assert {item.keep_split for item in plan} == {"val", "train"}
    near_item = next(item for item in plan if item.path == "test/a/test_near.png")
    assert near_item.pixel_area == 100
    assert near_item.keep_pixel_area == 600
    assert near_item.representative_keep_path == "train/a/train_near.png"
    assert summary["n_planned_deletions"] == 2

    results = apply_deletion_plan(data, plan)
    assert {item.status for item in results} == {"deleted"}
    assert not train_exact.exists()
    assert train_near.exists()
    assert val_exact.exists()
    assert not test_near.exists()


def test_connected_component_uses_one_keep_split(tmp_path):
    data = tmp_path / "data"
    paths = {
        "train/a/a.png": ((1, 1, 1), (40, 30)),
        "val/a/b.png": ((2, 2, 2), (20, 20)),
        "test/a/c.png": ((3, 3, 3), (10, 10)),
    }
    for relative, (color, size) in paths.items():
        _image(data / relative, color, size=size)
    for index in range(3):
        _image(data / "train" / "a" / f"extra_{index}.png", (10 + index,) * 3)
    _image(data / "val" / "a" / "extra.png", (20, 20, 20))

    report = tmp_path / "duplicates.csv"
    _write_duplicates(report, [
        {"duplicate_type": "near", "path_a": "train/a/a.png", "split_a": "train", "class_a": "a",
         "path_b": "val/a/b.png", "split_b": "val", "class_b": "a", "hamming_distance": "1",
         "cross_split": "True", "cross_class": "False"},
        {"duplicate_type": "near", "path_a": "val/a/b.png", "split_a": "val", "class_a": "a",
         "path_b": "test/a/c.png", "split_b": "test", "class_b": "a", "hamming_distance": "1",
         "cross_split": "True", "cross_class": "False"},
    ])

    plan, _ = build_deletion_plan(data, report)

    assert {item.keep_split for item in plan} == {"train"}
    assert {item.path for item in plan} == {"val/a/b.png", "test/a/c.png"}


def test_equal_dimensions_fall_back_to_smaller_split(tmp_path):
    data = tmp_path / "data"
    _image(data / "train" / "a" / "large_split.png", (1, 2, 3), size=(20, 10))
    _image(data / "val" / "a" / "small_split.png", (2, 3, 4), size=(20, 10))
    for index in range(3):
        _image(data / "train" / "a" / f"extra_{index}.png", (index,) * 3)
    report = tmp_path / "duplicates.csv"
    _write_duplicates(report, [{
        "duplicate_type": "near", "path_a": "train/a/large_split.png",
        "split_a": "train", "class_a": "a", "path_b": "val/a/small_split.png",
        "split_b": "val", "class_b": "a", "hamming_distance": "1",
        "cross_split": "True", "cross_class": "False",
    }])

    plan, _ = build_deletion_plan(data, report)

    assert len(plan) == 1
    assert plan[0].path == "train/a/large_split.png"
    assert plan[0].keep_split == "val"
    assert plan[0].selection_rule == "resolution_tie_then_smaller_split"


def test_max_near_distance_includes_two_three_and_four(tmp_path):
    data = tmp_path / "data"
    rows = []
    for distance in (1, 2, 3, 4):
        train_path = f"train/a/train_{distance}.png"
        val_path = f"val/a/val_{distance}.png"
        _image(data / train_path, (distance,) * 3, size=(30, 30))
        _image(data / val_path, (distance + 10,) * 3, size=(10, 10))
        rows.append({
            "duplicate_type": "near", "path_a": train_path,
            "split_a": "train", "class_a": "a", "path_b": val_path,
            "split_b": "val", "class_b": "a", "hamming_distance": str(distance),
            "cross_split": "True", "cross_class": "False",
        })
    report = tmp_path / "duplicates.csv"
    _write_duplicates(report, rows)

    default_plan, _ = build_deletion_plan(data, report)
    full_plan, summary = build_deletion_plan(data, report, max_near_distance=4)

    assert len(default_plan) == 1
    assert len(full_plan) == 4
    assert {item.path for item in full_plan} == {
        f"val/a/val_{distance}.png" for distance in (1, 2, 3, 4)
    }
    assert summary["near_duplicate_distance_range_removed"] == [1, 4]
