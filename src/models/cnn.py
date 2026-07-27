"""Các kiến trúc phân loại ảnh dùng trong pipeline.

MobileNetV3-Large được fine-tune toàn bộ ngay từ đầu. Backbone và classifier
dùng learning rate riêng nhưng cố định trong suốt quá trình huấn luyện.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

from src.utils.checkpoint import load_model_weights


BASELINE_ARCHITECTURES = (
    "mobilenetv3_small",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "swin_t",
    "vit_b_16",
)


def normalize_architecture_name(name: str) -> str:
    aliases = {
        "mobilenet_v3_small": "mobilenetv3_small",
        "efficientnet": "efficientnet_b0",
        "efficientnetb0": "efficientnet_b0",
        "swint": "swin_t",
        "vit": "vit_b_16",
        "vit_b16": "vit_b_16",
    }
    normalized = name.strip().lower().replace("-", "_")
    return aliases.get(normalized, normalized)


def _build_comparison_backbone(
    architecture: str,
    num_classes: int,
    pretrained: bool,
) -> Tuple[nn.Module, nn.Linear]:
    """Tạo torchvision model và trả về model cùng classifier cuối."""
    architecture = normalize_architecture_name(architecture)
    weights = "DEFAULT" if pretrained else None

    if architecture == "mobilenetv3_small":
        backbone = models.mobilenet_v3_small(weights=weights)
        old_head = backbone.classifier[-1]
        backbone.classifier[-1] = nn.Linear(old_head.in_features, num_classes)
        return backbone, backbone.classifier[-1]
    if architecture == "resnet50":
        backbone = models.resnet50(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        return backbone, backbone.fc
    if architecture == "densenet121":
        backbone = models.densenet121(weights=weights)
        backbone.classifier = nn.Linear(backbone.classifier.in_features, num_classes)
        return backbone, backbone.classifier
    if architecture == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=weights)
        old_head = backbone.classifier[-1]
        backbone.classifier[-1] = nn.Linear(old_head.in_features, num_classes)
        return backbone, backbone.classifier[-1]
    if architecture == "swin_t":
        backbone = models.swin_t(weights=weights)
        backbone.head = nn.Linear(backbone.head.in_features, num_classes)
        return backbone, backbone.head
    if architecture == "vit_b_16":
        backbone = models.vit_b_16(weights=weights)
        old_head = backbone.heads.head
        backbone.heads.head = nn.Linear(old_head.in_features, num_classes)
        return backbone, backbone.heads.head

    raise ValueError(
        f"Kiến trúc '{architecture}' không được hỗ trợ. "
        f"Các lựa chọn: {', '.join(BASELINE_ARCHITECTURES)}"
    )


class ImageClassificationBaseline(nn.Module):
    """Baseline đa kiến trúc, trả về ``(logits, penultimate_features)``.

    Hook chỉ đọc input của classifier cuối để lấy đặc trưng; nó không thay đổi
    forward/backward của torchvision model.
    """

    def __init__(
        self,
        architecture: str,
        num_classes: int = 12,
        pretrained: bool = True,
    ):
        super().__init__()
        self.architecture = normalize_architecture_name(architecture)
        self.backbone, self.head = _build_comparison_backbone(
            self.architecture, num_classes, pretrained
        )
        self._captured_features: Optional[torch.Tensor] = None
        self.head.register_forward_pre_hook(self._capture_head_input)

    def _capture_head_input(self, _module, inputs) -> None:
        self._captured_features = torch.flatten(inputs[0], 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.backbone(x)
        if self._captured_features is None:
            raise RuntimeError("Không lấy được đặc trưng đầu vào classifier.")
        features = self._captured_features
        self._captured_features = None
        return logits, features

    def trainable_param_groups(self, lr_head: float, lr_backbone: float):
        head_param_ids = {id(parameter) for parameter in self.head.parameters()}
        backbone_params = [
            parameter
            for parameter in self.backbone.parameters()
            if id(parameter) not in head_param_ids
        ]
        return [
            {"params": self.head.parameters(), "lr": lr_head, "name": "head"},
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
        ]


def _build_mobilenet_v3(num_classes: int) -> nn.Module:
    """Factory dùng chung để tạo backbone MobileNetV3-Large + thay classifier
    head — tránh lặp lại đoạn khởi tạo này ở cả CNNBaseline và FeatureExtractor."""
    backbone = models.mobilenet_v3_large(weights="DEFAULT")
    in_features = backbone.classifier[3].in_features
    backbone.classifier[3] = nn.Linear(in_features, num_classes)
    return backbone


class CNNBaseline(nn.Module):
    """MobileNetV3-Large làm backbone, trả về (logits, features 1280-d).

    Dùng cho cả baseline training (stage 1) lẫn rule-regularized fine-tuning
    (stage 5) lẫn inference serving (app/) — một class duy nhất, không có
    alias trùng tên.
    """

    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.backbone = _build_mobilenet_v3(num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.backbone.classifier[0:3](x)  # 1280-dim
        logits = self.backbone.classifier[3](features)
        return logits, features

    def trainable_param_groups(self, lr_head: float, lr_backbone: float):
        """Trả về hai nhóm tham số với learning rate cố định."""
        return [
            {"params": self.backbone.classifier.parameters(), "lr": lr_head, "name": "head"},
            {"params": self.backbone.features.parameters(), "lr": lr_backbone, "name": "backbone"},
        ]


class FeatureExtractor(nn.Module):
    """Backbone đã fine-tune, đóng băng toàn bộ, chỉ dùng để trích đặc trưng 1280-d."""

    def __init__(
        self,
        num_classes: int = 12,
        trained_model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.backbone = _build_mobilenet_v3(num_classes)
        if trained_model_path:
            load_model_weights(self, trained_model_path, device, required=False)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.backbone.classifier[0:3](x)
