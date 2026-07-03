"""Kiến trúc CNN + chiến lược transfer learning được chọn cho bài toán này.

CHIẾN LƯỢC ĐÃ CHỌN (xem giải thích đầy đủ trong README.md, mục "Transfer Learning"):
    "Freeze Backbone + Freeze BatchNorm"  ➜  "Differential / Layer-wise LR"
    ➜  "Progressive Unfreezing" theo giai đoạn (staged fine-tuning).

Lý do ngắn gọn: dataset ảnh văn hóa Việt Nam có domain lệch khá nhiều so với
ImageNet, kích thước dataset vừa/nhỏ, và các đặc trưng (features) trích ra từ
CNN này còn được downstream dùng để train Random Forest + GFlowNet — nên đặc
trưng cần ổn định (không "trôi" quá mạnh ngay từ epoch đầu). Vì vậy:
  Giai đoạn 1 (freeze_stage="head_only"): đóng băng toàn bộ backbone + BatchNorm,
      chỉ train classifier head → ổn định nhanh, tránh phá vỡ pretrained features.
  Giai đoạn 2 (freeze_stage="last_block"): mở block cuối của backbone với LR nhỏ
      hơn nhiều so với head (differential LR) → thích nghi domain mà không
      catastrophic forgetting.
  Giai đoạn 3 (freeze_stage="full"): mở toàn bộ mạng với layer-wise LR decay,
      dùng khi có đủ dữ liệu / cần accuracy tối đa.

GHI CHÚ: CHỈ MỘT class CNN duy nhất (`CNNBaseline`) được dùng xuyên suốt toàn
bộ pipeline (baseline training, rule-regularized fine-tuning, app serving).
Trước đây có alias `CNNWithFeatures` trùng lặp không cần thiết — đã gộp lại.
"""
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

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

    def __init__(self, num_classes: int = 12, freeze_stage: FreezeStage = "head_only"):
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
        self.freeze_bn()  # BatchNorm luôn được giữ đóng băng cho tới stage 'full'
        self._current_stage = stage

    def freeze_bn(self) -> None:
        """Đóng băng running stats + affine params của mọi BatchNorm2d.

        Quan trọng khi dataset/batch size nhỏ: BN dễ làm lệch running_mean/var
        và gây training không ổn định.
        """
        freeze = getattr(self, "_current_stage", "head_only") != "full"
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if freeze:
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                else:
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True

    def trainable_param_groups(self, lr_head: float, lr_backbone: float):
        """Trả về param groups cho optimizer, áp dụng Differential Learning Rate."""
        head_params = [p for p in self.backbone.classifier.parameters() if p.requires_grad]
        backbone_params = [p for p in self.backbone.features.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": lr_head}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
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
        if trained_model_path and os.path.exists(trained_model_path):
            state_dict = torch.load(trained_model_path, map_location=device)
            self.load_state_dict(state_dict)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.backbone.classifier[0:3](x)
