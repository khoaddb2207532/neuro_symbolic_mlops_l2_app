"""Xoa logo co vi tri co dinh bang cach thay bang mot patch nen lan can.

Thay vi phu mot hinh chu nhat den (co the tro thanh shortcut moi), module lay
mot patch cung kich thuoc o ngay ben duoi/ben trai logo, dieu chinh mau trung
binh nhe va feather bien khi ghep. Anh nguon khong bao gio bi ghi de.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from src.data.dataset import VALID_EXTENSIONS


@dataclass
class LogoRemovalRecord:
    source_path: str
    output_path: str
    width: int
    height: int
    logo_x1: int
    logo_y1: int
    logo_x2: int
    logo_y2: int
    patch_x1: int
    patch_y1: int
    patch_x2: int
    patch_y2: int
    source_sha256: str
    output_sha256: str = ""
    status: str = "planned"
    error: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_roi_to_pixels(
    size: Tuple[int, int], roi: Sequence[float]
) -> Tuple[int, int, int, int]:
    if len(roi) != 4:
        raise ValueError("ROI must contain x1 y1 x2 y2")
    x1n, y1n, x2n, y2n = (float(value) for value in roi)
    if not (0 <= x1n < x2n <= 1 and 0 <= y1n < y2n <= 1):
        raise ValueError("Normalized ROI values must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    width, height = size
    x1 = max(0, min(width - 1, round(x1n * width)))
    y1 = max(0, min(height - 1, round(y1n * height)))
    x2 = max(x1 + 1, min(width, round(x2n * width)))
    y2 = max(y1 + 1, min(height, round(y2n * height)))
    return x1, y1, x2, y2


def _choose_neighbor_patch(
    image_size: Tuple[int, int], logo_box: Tuple[int, int, int, int], gap: int
) -> Tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = logo_box
    box_width, box_height = x2 - x1, y2 - y1

    # Uu tien patch ngay ben duoi vi logo thuong o sat mep tren/phai.
    below_y1 = y2 + gap
    if below_y1 + box_height <= height:
        return x1, below_y1, x2, below_y1 + box_height

    left_x1 = x1 - gap - box_width
    if left_x1 >= 0:
        return left_x1, y1, left_x1 + box_width, y2

    above_y1 = y1 - gap - box_height
    if above_y1 >= 0:
        return x1, above_y1, x2, above_y1 + box_height

    raise ValueError("Image is too small to find a neighboring replacement patch")


def _boundary_interpolation_fill(
    image: Image.Image,
    logo_box: Tuple[int, int, int, int],
    texture_patch: Image.Image,
) -> Image.Image:
    """Noi suy tu bon bien va them high-frequency texture nhe tu patch lan can."""
    x1, y1, x2, y2 = logo_box
    box_width, box_height = x2 - x1, y2 - y1
    array = np.asarray(image, dtype=np.float32)

    top = array[max(0, y1 - 1), x1:x2]
    bottom = array[min(image.height - 1, y2), x1:x2]
    left = array[y1:y2, max(0, x1 - 1)]
    right = array[y1:y2, min(image.width - 1, x2)]

    ty = np.linspace(0, 1, box_height, dtype=np.float32)[:, None, None]
    tx = np.linspace(0, 1, box_width, dtype=np.float32)[None, :, None]
    vertical = (1 - ty) * top[None, :, :] + ty * bottom[None, :, :]
    horizontal = (1 - tx) * left[:, None, :] + tx * right[:, None, :]
    fill = 0.5 * vertical + 0.5 * horizontal

    texture_array = np.asarray(texture_patch, dtype=np.float32)
    smooth_texture = np.asarray(
        texture_patch.filter(ImageFilter.GaussianBlur(radius=4)), dtype=np.float32
    )
    high_frequency = texture_array - smooth_texture
    fill = np.clip(fill + 0.18 * high_frequency, 0, 255).astype(np.uint8)
    return Image.fromarray(fill, mode="RGB")


def remove_logo(
    image: Image.Image,
    normalized_roi: Sequence[float] = (0.815, 0.035, 0.99, 0.15),
    feather_radius: int = 3,
    neighbor_gap: int = 4,
) -> Tuple[Image.Image, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    logo_box = normalized_roi_to_pixels(image.size, normalized_roi)
    patch_box = _choose_neighbor_patch(image.size, logo_box, neighbor_gap)
    texture_patch = image.crop(patch_box)
    patch = _boundary_interpolation_fill(image, logo_box, texture_patch)

    mask = Image.new("L", patch.size, color=255)
    if feather_radius > 0:
        # Alpha bang 255 o phan lon ROI, chi giam trong mot vien mong.
        yy, xx = np.indices((patch.height, patch.width))
        distance_to_edge = np.minimum.reduce(
            (xx, yy, patch.width - 1 - xx, patch.height - 1 - yy)
        ).astype(np.float32)
        alpha = np.clip((distance_to_edge + 1) / max(feather_radius, 1), 0, 1)
        alpha = (alpha * 255).astype(np.uint8)
        mask = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=0.6))

    output = image.copy()
    output.paste(patch, (logo_box[0], logo_box[1]), mask)
    return output, logo_box, patch_box


def iter_images(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VALID_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        yield input_path
        return
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    yield from (
        path for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def process_logo_removal(
    input_path: Path,
    output_dir: Path,
    normalized_roi: Sequence[float] = (0.815, 0.035, 0.99, 0.15),
    feather_radius: int = 3,
    neighbor_gap: int = 4,
    quality: int = 95,
) -> List[LogoRemovalRecord]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if input_path.is_dir():
        relative_root = input_path
    else:
        relative_root = input_path.parent
    if output_dir == input_path or output_dir == relative_root:
        raise ValueError("Output directory must differ from the source directory")

    records: List[LogoRemovalRecord] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in iter_images(input_path):
        relative = source.relative_to(relative_root)
        destination = output_dir / relative
        source_hash = sha256_file(source)
        try:
            with Image.open(source) as image:
                result, logo_box, patch_box = remove_logo(
                    image, normalized_roi, feather_radius, neighbor_gap
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs = {}
            if destination.suffix.lower() in (".jpg", ".jpeg"):
                save_kwargs = {"quality": quality, "subsampling": 0, "optimize": True}
            result.save(destination, **save_kwargs)
            record = LogoRemovalRecord(
                source_path=str(source), output_path=str(destination),
                width=result.width, height=result.height,
                logo_x1=logo_box[0], logo_y1=logo_box[1],
                logo_x2=logo_box[2], logo_y2=logo_box[3],
                patch_x1=patch_box[0], patch_y1=patch_box[1],
                patch_x2=patch_box[2], patch_y2=patch_box[3],
                source_sha256=source_hash, output_sha256=sha256_file(destination),
                status="processed",
            )
        except Exception as exc:
            record = LogoRemovalRecord(
                source_path=str(source), output_path=str(destination),
                width=0, height=0,
                logo_x1=0, logo_y1=0, logo_x2=0, logo_y2=0,
                patch_x1=0, patch_y1=0, patch_x2=0, patch_y2=0,
                source_sha256=source_hash, status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        records.append(record)
    return records


def write_logo_report(path: Path, records: Sequence[LogoRemovalRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(LogoRemovalRecord.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
