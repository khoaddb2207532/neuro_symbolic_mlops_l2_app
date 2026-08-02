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
    "alexnet",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "swin_t",
    "vit_b_16",
    "vit_b_32",
)


def normalize_architecture_name(name: str) -> str:
    aliases = {
        "mobilenet_v3_small": "mobilenetv3_small",
        "efficientnet": "efficientnet_b0",
        "efficientnetb0": "efficientnet_b0",
        "swint": "swin_t",
        "swinti": "swin_t",
        "swin_tiny": "swin_t",
        "alex": "alexnet",
        "vit": "vit_b_32",
        "vit_b16": "vit_b_16",
        "vit_b32": "vit_b_32",
    }
    normalized = name.strip().lower().replace("-", "_")
    return aliases.get(normalized, normalized)


class _FeatureClassifierBackbone(nn.Module):
    """Ghép backbone trả feature với classifier mới, giữ interface torchvision."""

    def __init__(
        self,
        feature_backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.feature_backbone = feature_backbone
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.feature_backbone(x))


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
    if architecture == "alexnet":
        backbone = models.alexnet(weights=weights)
        old_head = backbone.classifier[-1]
        backbone.classifier[-1] = nn.Linear(old_head.in_features, num_classes)
        # Forward-pre-hook trên lớp cuối nhận đúng feature sau classifier[:-1].
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
        swin = models.swin_t(weights=weights)
        feature_dim = swin.head.in_features
        swin.head = nn.Identity()
        backbone = _FeatureClassifierBackbone(swin, feature_dim, num_classes)
        return backbone, backbone.fc
    if architecture == "vit_b_16":
        vit = models.vit_b_16(weights=weights)
        feature_dim = vit.hidden_dim
        vit.heads = nn.Identity()
        backbone = _FeatureClassifierBackbone(vit, feature_dim, num_classes)
        return backbone, backbone.fc
    if architecture == "vit_b_32":
        vit = models.vit_b_32(weights=weights)
        feature_dim = vit.hidden_dim
        vit.heads = nn.Identity()
        backbone = _FeatureClassifierBackbone(vit, feature_dim, num_classes)
        return backbone, backbone.fc

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
        freeze_features: bool = False,
    ):
        super().__init__()
        self.architecture = normalize_architecture_name(architecture)
        self.backbone, self.head = _build_comparison_backbone(
            self.architecture, num_classes, pretrained
        )
        self._captured_features: Optional[torch.Tensor] = None
        self.head.register_forward_pre_hook(self._capture_head_input)
        if freeze_features:
            head_param_ids = {id(parameter) for parameter in self.head.parameters()}
            for parameter in self.backbone.parameters():
                if id(parameter) not in head_param_ids:
                    parameter.requires_grad = False

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
            if id(parameter) not in head_param_ids and parameter.requires_grad
        ]
        return [
            {
                "params": [
                    parameter for parameter in self.head.parameters()
                    if parameter.requires_grad
                ],
                "lr": lr_head,
                "name": "head",
            },
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
        ]

    def freeze_bn(self) -> None:
        """Đóng băng BatchNorm của CNN và LayerNorm của Swin/ViT."""
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()
                if module.weight is not None:
                    module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False


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
