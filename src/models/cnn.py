"""MobileNetV3-Large khởi tạo từ ImageNet để huấn luyện end-to-end.

Trainer chính cập nhật toàn bộ backbone và classifier ngay từ epoch đầu tiên
với một learning rate; riêng BatchNorm được khóa từ đầu đến cuối. Các
``freeze_stage`` cũ vẫn được giữ để tương thích với checkpoint và các thí
nghiệm ablation hiện có.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

from src.utils.checkpoint import load_model_weights

FreezeStage = str  # "head_only" | "last_block" | "full"


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
        
