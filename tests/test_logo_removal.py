from pathlib import Path

import numpy as np
from PIL import Image

from src.data.logo_removal import process_logo_removal, remove_logo


def test_remove_logo_changes_roi_but_not_source():
    array = np.zeros((100, 200, 3), dtype=np.uint8)
    array[:] = (30, 70, 100)
    array[5:20, 160:195] = (255, 255, 255)
    image = Image.fromarray(array, mode="RGB")

    output, logo_box, _ = remove_logo(
        image, normalized_roi=(0.8, 0.05, 0.975, 0.2), feather_radius=2
    )

    original = np.asarray(image)
    cleaned = np.asarray(output)
    x1, y1, x2, y2 = logo_box
    assert np.abs(cleaned[y1:y2, x1:x2].astype(int) - original[y1:y2, x1:x2]).mean() > 50
    assert np.array_equal(cleaned[50:80, 20:80], original[50:80, 20:80])


def test_batch_processing_preserves_source_and_writes_output(tmp_path):
    source_dir = tmp_path / "source"
    source = source_dir / "class-a" / "sample.jpg"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (200, 100), color=(20, 40, 60)).save(source)
    before = source.read_bytes()
    output_dir = tmp_path / "cleaned"

    records = process_logo_removal(source_dir, output_dir)

    assert len(records) == 1
    assert records[0].status == "processed"
    assert (output_dir / "class-a" / "sample.jpg").is_file()
    assert source.read_bytes() == before
