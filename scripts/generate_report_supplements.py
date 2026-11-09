"""Generate deterministic dataset sample grids and posterior summaries for the report."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "report" / "figures"
GENERATED_DIR = ROOT / "report" / "generated"
DATASET_A = Path(r"D:\32526_LV\vietnamese_cultural_dataset\train")
DATASET_B = Path(r"D:\Downloads\Fast Food Classification V2\Train")
POSTERIOR_ROOT = ROOT / "tmp" / "bundle_audit"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LABELS_A = {
    "cho-noi-cai-rang": "Chợ nổi Cái Răng",
    "chua": "Chùa",
    "dieu-mua-truyen-thong": "Điệu múa truyền thống",
    "don-ca-tai-tu": "Đờn ca tài tử",
    "dua-bo-bay-nui": "Đua bò Bảy Núi",
    "le-hoi-nghinh-ong": "Lễ hội Nghinh Ông",
    "nghe-dan-tre": "Nghề đan tre",
    "nghe-det-chieu": "Nghề dệt chiếu",
    "nghe-thuat-san-khau-du-ke-khmer": "Sân khấu Dù Kê Khmer",
    "ok-om-bok": "Ok Om Bok",
    "phong-tuc-cuoi-truyen-thong": "Phong tục cưới truyền thống",
    "via-ba-chua-xu-nui-sam": "Vía Bà Chúa Xứ Núi Sam",
}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def representative_image(class_dir: Path) -> Path:
    files = sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No image found in {class_dir}")
    # The median filename is deterministic and avoids hand-picking examples.
    return files[len(files) // 2]


def make_grid(
    source: Path,
    output: Path,
    columns: int,
    label_map: dict[str, str] | None = None,
    font_size: int = 30,
) -> None:
    class_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    rows = (len(class_dirs) + columns - 1) // columns
    cell_width, image_height, label_height = 360, 240, 48
    gap, margin = 14, 18
    canvas_width = margin * 2 + columns * cell_width + (columns - 1) * gap
    canvas_height = margin * 2 + rows * (image_height + label_height) + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(font_size)

    for index, class_dir in enumerate(class_dirs):
        row, column = divmod(index, columns)
        x = margin + column * (cell_width + gap)
        y = margin + row * (image_height + label_height + gap)
        path = representative_image(class_dir)
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            image = ImageOps.fit(image, (cell_width, image_height), method=Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + cell_width - 1, y + image_height - 1), outline="#444444", width=2)
        label = label_map.get(class_dir.name, class_dir.name) if label_map else class_dir.name
        label_font_size = font_size
        label_font = font
        box = draw.textbbox((0, 0), label, font=label_font)
        while box[2] - box[0] > cell_width - 12 and label_font_size > 18:
            label_font_size -= 1
            label_font = load_font(label_font_size)
            box = draw.textbbox((0, 0), label, font=label_font)
        text_width = box[2] - box[0]
        text_x = x + max(4, (cell_width - text_width) // 2)
        text_y = y + image_height + 8
        draw.text((text_x, text_y), label, fill="black", font=label_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return (sum((value - center) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def aggregate_posterior(output: Path) -> None:
    metrics = (
        "mean_log_reward_mean",
        "unique_ratio_mean",
        "entropy_normalized_mean",
        "top1_share_mean",
        "mean_ruleset_size_mean",
    )
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    pattern = "runs/*/*_checkpoint_posterior_evaluation_summary.csv"
    for path in sorted(POSTERIOR_ROOT.glob(f"bundle_*/{pattern}")):
        run_id = path.parent.name
        dataset_id = "culture-a" if run_id.startswith("culture-a__") else "culture-b"
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                groups.setdefault((dataset_id, row["checkpoint"]), []).append(row)

    if len(groups) != 4 or any(len(rows) != 25 for rows in groups.values()):
        counts = {key: len(rows) for key, rows in groups.items()}
        raise ValueError(f"Expected four groups of 25 rows, received {counts}")

    fields = ["dataset_id", "checkpoint", "n_runs"]
    for metric in metrics:
        fields.extend((f"{metric}_across_runs", f"{metric}_std_across_runs"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset_id, checkpoint in sorted(groups):
            rows = groups[(dataset_id, checkpoint)]
            result: dict[str, str | int | float] = {
                "dataset_id": dataset_id,
                "checkpoint": checkpoint,
                "n_runs": len(rows),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in rows]
                result[f"{metric}_across_runs"] = mean(values)
                result[f"{metric}_std_across_runs"] = sample_std(values)
            writer.writerow(result)


def main() -> None:
    make_grid(DATASET_A, FIGURE_DIR / "dataset-a-samples.png", columns=4, label_map=LABELS_A)
    make_grid(DATASET_B, FIGURE_DIR / "dataset-b-samples.png", columns=5, font_size=34)
    aggregate_posterior(GENERATED_DIR / "posterior-diversity-summary.csv")


if __name__ == "__main__":
    main()
