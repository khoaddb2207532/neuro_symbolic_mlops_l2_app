"""MobileNetV3-Large khởi tạo từ ImageNet để huấn luyện end-to-end.

Trainer chính cập nhật toàn bộ backbone và classifier ngay từ epoch đầu tiên
với một learning rate; riêng BatchNorm được khóa từ đầu đến cuối. Các
``freeze_stage`` cũ vẫn được giữ để tương thích với checkpoint và các thí
nghiệm ablation hiện có.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

from src.utils.checkpoint import load_model_weights

FreezeStage = str  # "head_only" | "last_block" | "full"

SUPPORTED_BASELINES = (
    "mobilenetv3_small",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "swin_t",
    "vit_b_16",
)

BASELINE_ALIASES: Dict[str, str] = {
    "efficientnet": "efficientnet_b0",
    "efficientnetb0": "efficientnet_b0",
    "effcientnet": "efficientnet_b0",  # common misspelling in experiment configs
    "swint": "swin_t",
    "vit": "vit_b_16",
}


def canonical_baseline_name(name: str) -> str:
    """Return the canonical torchvision name used in output paths and reports."""
    normalized = name.strip().lower().replace("-", "_")
    normalized = BASELINE_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_BASELINES:
        raise ValueError(
            f"Unsupported baseline '{name}'. Choose one of: {', '.join(SUPPORTED_BASELINES)}"
        )
    return normalized


class VisionBaseline(nn.Module):
    """ImageNet-pretrained torchvision baseline returning ``(logits, features)``.

    The common interface makes CNN and Transformer backbones directly compatible
    with the existing trainer, evaluator, and future feature-based stages.
    """

    def __init__(self, architecture: str, num_classes: int = 12, pretrained: bool = True):
        super().__init__()
        self.architecture = canonical_baseline_name(architecture)
        weights = "DEFAULT" if pretrained else None

        if self.architecture == "mobilenetv3_small":
            self.backbone = models.mobilenet_v3_small(weights=weights)
            self.feature_dim = self.backbone.classifier[0].out_features
            self.backbone.classifier[-1] = nn.Linear(self.feature_dim, num_classes)
        elif self.architecture == "resnet50":
            self.backbone = models.resnet50(weights=weights)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(self.feature_dim, num_classes)
        elif self.architecture == "densenet121":
            self.backbone = models.densenet121(weights=weights)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(self.feature_dim, num_classes)
        elif self.architecture == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=weights)
            self.feature_dim = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(self.feature_dim, num_classes)
        elif self.architecture == "swin_t":
            self.backbone = models.swin_t(weights=weights)
            self.feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Linear(self.feature_dim, num_classes)
        else:  # vit_b_16
            self.backbone = models.vit_b_16(weights=weights)
            self.feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        architecture = self.architecture
        if architecture in {"mobilenetv3_small", "efficientnet_b0"}:
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            features = self.backbone.classifier[:-1](x)
            logits = self.backbone.classifier[-1](features)
        elif architecture == "resnet50":
            b = self.backbone
            x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
            x = b.layer1(x); x = b.layer2(x); x = b.layer3(x); x = b.layer4(x)
            features = torch.flatten(b.avgpool(x), 1)
            logits = b.fc(features)
        elif architecture == "densenet121":
            x = self.backbone.features(x)
            features = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(
                torch.nn.functional.relu(x, inplace=True), (1, 1)
            ), 1)
            logits = self.backbone.classifier(features)
        elif architecture == "swin_t":
            b = self.backbone
            x = b.features(x); x = b.norm(x); x = b.permute(x); x = b.avgpool(x)
            features = b.flatten(x)
            logits = b.head(features)
        else:
            b = self.backbone
            x = b._process_input(x)
            class_token = b.class_token.expand(x.shape[0], -1, -1)
            features = b.encoder(torch.cat([class_token, x], dim=1))[:, 0]
            logits = b.heads(features)
        return logits, features

    def freeze_bn(self) -> None:
        """Keep BatchNorm statistics fixed, matching the existing baseline policy."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                if module.affine:
                    module.weight.requires_grad = False
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

    def __init__(self, num_classes: int = 12, freeze_stage: FreezeStage = "full"):
        super().__init__()
        self.backbone = _build_mobilenet_v3(num_classes)
        self.set_freeze_stage(freeze_stage)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.backbone.classifier[0:3](x)  # 1280-dim
        logits = self.backbone.classifier[3](features)
        return logits, features

    # ------------------------------------------------------------------ #
    #  Chiến lược transfer learning theo giai đoạn (progressive unfreeze) #
    # ------------------------------------------------------------------ #
    def set_freeze_stage(self, stage: FreezeStage) -> None:
        """Áp dụng đóng băng theo giai đoạn được chọn. Gọi lại mỗi khi chuyển stage."""
        if stage == "head_only":
            for p in self.backbone.features.parameters():
                p.requires_grad = False
            for p in self.backbone.classifier.parameters():
                p.requires_grad = True
        elif stage == "last_block":
            n_feature_blocks = len(self.backbone.features)
            last_block_start = max(0, n_feature_blocks - 3)  # 3 block cuối
            for i, block in enumerate(self.backbone.features):
                requires_grad = i >= last_block_start
                for p in block.parameters():
                    p.requires_grad = requires_grad
            for p in self.backbone.classifier.parameters():
                p.requires_grad = True
        elif stage == "full":
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"freeze_stage phải là 'head_only'|'last_block'|'full', nhận '{stage}'")
        self._current_stage = stage
        self.freeze_bn()

    def freeze_bn(self) -> None:
        """Khóa running statistics và affine params của mọi BatchNorm2d."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def trainable_param_groups(self, lr_head: float, lr_backbone: float):
        """Trả về param groups cho optimizer, áp dụng Differential Learning Rate."""
        head_params = [p for p in self.backbone.classifier.parameters() if p.requires_grad]
        backbone_params = [p for p in self.backbone.features.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": lr_head, "name": "head"}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone, "name" : "backbone"})
        return groups


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
        features = self.backbone.classifier[0:3](x)  # 1280-dim
        logits = self.backbone.classifier[3](features)
        return logits, features
        
