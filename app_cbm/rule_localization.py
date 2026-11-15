"""Spatial explanations for rules defined on global CNN features.

The rules do not contain image coordinates.  We therefore localize them with
occlusion sensitivity: hide one grid cell at a time and measure how much the
rule satisfaction drops.  Connected influential cells become bounding boxes.
This is deliberately rule-aware and is not Grad-CAM.
"""
from __future__ import annotations

import base64
import io
from collections import deque
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD


RULE_COLORS: Sequence[Tuple[int, int, int]] = (
    (37, 99, 235),
    (161, 98, 7),
    (5, 150, 105),
    (220, 38, 38),
    (124, 58, 237),
)


def model_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert the normalized 3xHxW model input back to the visible crop."""
    image = tensor.detach().cpu().clone()
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
    return to_pil_image((image * std + mean).clamp(0, 1))


def extract_bounding_boxes(
    importance: torch.Tensor,
    image_size: Tuple[int, int],
    relative_threshold: float = 0.35,
    max_regions: int = 2,
) -> List[Dict[str, float]]:
    """Turn a non-negative grid importance map into connected image boxes."""
    if importance.ndim != 2:
        raise ValueError("importance must be a 2D grid")

    values = importance.detach().float().cpu().clamp_min(0)
    peak = float(values.max().item()) if values.numel() else 0.0
    if peak <= 1e-6:
        return []

    positive = values[values > 0]
    # The relative peak threshold suppresses numerical noise.  A softened
    # quantile keeps adjacent, slightly weaker cells in the same region.
    quantile = float(torch.quantile(positive, 0.60).item())
    threshold = max(peak * relative_threshold, quantile * 0.5)
    active = values >= threshold
    rows, cols = active.shape
    visited = torch.zeros_like(active, dtype=torch.bool)
    components = []

    for row in range(rows):
        for col in range(cols):
            if not bool(active[row, col]) or bool(visited[row, col]):
                continue
            queue = deque([(row, col)])
            visited[row, col] = True
            cells = []
            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if (
                        0 <= next_row < rows
                        and 0 <= next_col < cols
                        and bool(active[next_row, next_col])
                        and not bool(visited[next_row, next_col])
                    ):
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
            strength = sum(float(values[r, c].item()) for r, c in cells)
            components.append((strength, cells))

    width, height = image_size
    boxes: List[Dict[str, float]] = []
    for strength, cells in sorted(components, reverse=True)[:max_regions]:
        min_row = min(r for r, _ in cells)
        max_row = max(r for r, _ in cells) + 1
        min_col = min(c for _, c in cells)
        max_col = max(c for _, c in cells) + 1
        boxes.append(
            {
                "x1": int(round(min_col * width / cols)),
                "y1": int(round(min_row * height / rows)),
                "x2": int(round(max_col * width / cols)),
                "y2": int(round(max_row * height / rows)),
                "importance": round(min(1.0, strength / (peak * len(cells))), 4),
            }
        )
    return boxes


def draw_rule_boxes(image: Image.Image, rule_regions: List[Dict]) -> Image.Image:
    """Draw stable, accessible rule colors plus text labels on an image."""
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for region in rule_regions:
        color = tuple(region["color"])
        for box in region["boxes"]:
            coordinates = (box["x1"], box["y1"], box["x2"], box["y2"])
            overlay_draw.rectangle(coordinates, fill=(*color, 35))
    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, min(canvas.size) // 90)

    for region in rule_regions:
        color = tuple(region["color"])
        label = f"Rule {region['rule_number']} | {region['satisfaction_score'] * 100:.0f}%"
        for box in region["boxes"]:
            coordinates = (box["x1"], box["y1"], box["x2"], box["y2"])
            draw.rectangle(coordinates, outline=(*color, 255), width=line_width)
            text_box = draw.textbbox((0, 0), label)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            label_y = max(0, box["y1"] - text_height - 8)
            draw.rounded_rectangle(
                (box["x1"], label_y, box["x1"] + text_width + 10, label_y + text_height + 8),
                radius=4,
                fill=(*color, 255),
            )
            draw.text((box["x1"] + 5, label_y + 3), label, fill=(255, 255, 255, 255))
    return canvas.convert("RGB")


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def base64_to_image(value: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
