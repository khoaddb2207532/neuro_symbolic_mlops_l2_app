"""Concept Bottleneck Model (CBM) cho pipeline neuro-symbolic.

Ý tưởng cốt lõi:
  Backbone CNN → Global Avg Pool → Flatten → **Concept Layer** (sigmoid)
  → Label Predictor (Linear)

Thay vì vector đặc trưng 1280-d (MobileNetV3) hay 768-d (ViT), CBM ép
feature vector xuống còn `n_concepts` chiều, mỗi chiều đại diện cho xác suất
xuất hiện của một concept (ví dụ: "mây", "nón lá", "sông", "thuyền", "áo dài",
...).  Sigmoid đảm bảo mỗi giá trị nằm trong [0, 1].

Để các concept score thực sự phản ánh đúng xác suất xuất hiện của concept,
trong quá trình huấn luyện ta thêm **concept loss** (Binary Cross-Entropy giữa
concept predictions và concept annotations do người gán) bên cạnh task loss
(Cross-Entropy cho label classification).

Trade-off:
  - Giải thích được: truy ngược concept scores để xem mô hình "thấy gì".
  - Mất mát thông tin: n_concepts << feature_dim gốc → có thể giảm acc/F1.
  - Số lượng concept KHÔNG nên quá nhiều (loãng, concept nào cũng xuất hiện →
    mất tính phân biệt) cũng không quá ít (thiếu thông tin).
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

from src.models.cnn import (
    BASELINE_ARCHITECTURES,
    normalize_architecture_name,
)
from src.utils.checkpoint import load_model_weights


# ---------------------------------------------------------------------------
# Hàm tiện ích: lấy feature dimension của từng backbone
# ---------------------------------------------------------------------------

def _get_backbone_feature_dim(architecture: str) -> int:
    """Trả về số chiều đặc trưng (sau global avg pool / flatten) của backbone."""
    dims = {
        "mobilenetv3_small": 576,
        "shufflenet_v2_x1_0": 1024,
        "alexnet": 4096,       # sau classifier[:-1]
        "resnet50": 2048,
        "densenet121": 1024,
        "efficientnet_b0": 1280,
        "swin_t": 768,
        "vit_b_16": 768,
        "vit_b_32": 768,
    }
    return dims.get(architecture, 1280)


# ---------------------------------------------------------------------------
# Backbone feature extractor (đóng băng hoặc fine-tune, trả raw features)
# ---------------------------------------------------------------------------

class _BackboneFeatures(nn.Module):
    """Trích raw feature vector từ backbone torchvision (trước classifier cuối).

    Forward trả về tensor [B, feature_dim] — vector TRƯỚC khi đi qua
    classifier gốc.
    """

    def __init__(self, architecture: str, pretrained: bool = True):
        super().__init__()
        self.architecture = normalize_architecture_name(architecture)
        weights = "DEFAULT" if pretrained else None
        self.feature_dim: int

        if self.architecture == "mobilenetv3_small":
            bb = models.mobilenet_v3_small(weights=weights)
            # features → avgpool → flatten → classifier[0:2] (Linear+HardSwish)
            # Nhưng CBM muốn raw features SAU avgpool, TRƯỚC classifier head.
            self._features = bb.features
            self._avgpool = bb.avgpool
            self.feature_dim = 576

        elif self.architecture == "shufflenet_v2_x1_0":
            bb = models.shufflenet_v2_x1_0(weights=weights)
            self._features = nn.Sequential(
                bb.conv1, bb.maxpool, bb.stage2, bb.stage3, bb.stage4, bb.conv5,
            )
            self._avgpool = nn.AdaptiveAvgPool2d(1)
            self.feature_dim = 1024

        elif self.architecture == "alexnet":
            bb = models.alexnet(weights=weights)
            self._features = bb.features
            self._avgpool = bb.avgpool
            # alexnet.classifier = [Dropout, Linear(9216,4096), ReLU,
            #   Dropout, Linear(4096,4096), ReLU, Linear(4096,1000)]
            # Ta lấy output sau classifier[0:5] = 4096-d
            self._fc_prefix = bb.classifier[0:5]
            self.feature_dim = 4096

        elif self.architecture == "resnet50":
            bb = models.resnet50(weights=weights)
            self._features = nn.Sequential(
                bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                bb.layer1, bb.layer2, bb.layer3, bb.layer4,
            )
            self._avgpool = bb.avgpool
            self.feature_dim = 2048

        elif self.architecture == "densenet121":
            bb = models.densenet121(weights=weights)
            self._features = bb.features
            self._avgpool = nn.AdaptiveAvgPool2d(1)
            self.feature_dim = 1024

        elif self.architecture == "efficientnet_b0":
            bb = models.efficientnet_b0(weights=weights)
            self._features = bb.features
            self._avgpool = bb.avgpool
            self.feature_dim = 1280

        elif self.architecture == "swin_t":
            bb = models.swin_t(weights=weights)
            bb.head = nn.Identity()
            self._features = bb
            self._avgpool = nn.Identity()
            self.feature_dim = 768

        elif self.architecture in ("vit_b_16", "vit_b_32"):
            bb = (models.vit_b_16 if self.architecture == "vit_b_16"
                  else models.vit_b_32)(weights=weights)
            bb.heads = nn.Identity()
            self._features = bb
            self._avgpool = nn.Identity()
            self.feature_dim = 768

        else:
            raise ValueError(
                f"Kiến trúc '{architecture}' không được hỗ trợ cho CBM. "
                f"Các lựa chọn: {', '.join(BASELINE_ARCHITECTURES)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x)
        x = self._avgpool(x)
        x = torch.flatten(x, 1)
        # AlexNet cần thêm bước qua fc_prefix
        if hasattr(self, "_fc_prefix"):
            x = self._fc_prefix(x)
        return x


# ---------------------------------------------------------------------------
# Concept Bottleneck Model
# ---------------------------------------------------------------------------

class ConceptBottleneckModel(nn.Module):
    """Concept Bottleneck Model (CBM).

    Kiến trúc::

        Input Image
            │
            ▼
        Backbone (pretrained, fine-tune hoặc freeze)
            │  → raw features [B, feature_dim]
            ▼
        Concept Layer (Linear + Sigmoid)
            │  → concept_probs [B, n_concepts] ∈ [0, 1]
            ▼
        Label Predictor (Linear)
            │  → logits [B, num_classes]
            ▼
        Output: (logits, concept_probs, raw_features)

    Huấn luyện:
        total_loss = task_loss(logits, labels)
                   + concept_weight * concept_loss(concept_probs, concept_targets)

        trong đó concept_loss = BCELoss (vì concept_probs đã qua sigmoid).

    Attributes:
        n_concepts: Số lượng concepts (chiều của bottleneck).
        concept_names: Tên từng concept (dùng khi visualize).
    """

    def __init__(
        self,
        architecture: str = "mobilenetv3_small",
        n_concepts: int = 30,
        num_classes: int = 12,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        concept_names: Optional[list] = None,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.num_classes = num_classes
        self.concept_names = concept_names or [f"concept_{i}" for i in range(n_concepts)]

        # Backbone: trích feature vector
        self.backbone = _BackboneFeatures(architecture, pretrained=pretrained)
        feature_dim = self.backbone.feature_dim

        # Concept layer: feature_dim → n_concepts, sigmoid ép về [0, 1]
        self.concept_layer = nn.Sequential(
            nn.Linear(feature_dim, n_concepts),
            nn.Sigmoid(),
        )

        # Label predictor: n_concepts → num_classes
        # Đây là "head" cuối cùng, input chỉ là concept probabilities
        self.label_predictor = nn.Linear(n_concepts, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            logits: [B, num_classes] — logits phân loại.
            concept_probs: [B, n_concepts] — xác suất từng concept ∈ [0,1].
            raw_features: [B, feature_dim] — đặc trưng gốc từ backbone.
        """
        raw_features = self.backbone(x)
        concept_probs = self.concept_layer(raw_features)
        logits = self.label_predictor(concept_probs)
        return logits, concept_probs, raw_features

    def trainable_param_groups(self, lr_head: float, lr_backbone: float):
        """Trả về 3 nhóm tham số với learning rate riêng.

        - backbone: lr_backbone (thường thấp hơn vì pretrained)
        - concept_layer: lr_head
        - label_predictor: lr_head
        """
        backbone_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        return [
            {"params": list(self.concept_layer.parameters()) + list(self.label_predictor.parameters()),
             "lr": lr_head, "name": "head"},
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
        ]

    def freeze_bn(self) -> None:
        """Đóng băng BatchNorm/LayerNorm."""
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()
                if module.weight is not None:
                    module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False

    def explain(self, x: torch.Tensor, top_k: int = 5) -> list:
        """Giải thích dự đoán: top-k concepts có xác suất cao nhất.

        Args:
            x: Ảnh đầu vào [1, C, H, W] hoặc [B, C, H, W].
            top_k: Số concept cần trả về.

        Returns:
            List[dict] — mỗi dict chứa {'concept': str, 'probability': float}.
        """
        self.eval()
        with torch.no_grad():
            _, concept_probs, _ = self.forward(x)

        results = []
        for i in range(concept_probs.size(0)):
            probs = concept_probs[i]
            top_values, top_indices = torch.topk(probs, min(top_k, self.n_concepts))
            explanations = [
                {
                    "concept": self.concept_names[idx.item()],
                    "probability": val.item(),
                }
                for val, idx in zip(top_values, top_indices)
            ]
            results.append(explanations)
        return results


# ---------------------------------------------------------------------------
# Factory: tạo CBM từ baseline checkpoint đã train
# ---------------------------------------------------------------------------

def create_cbm_from_baseline(
    architecture: str,
    n_concepts: int,
    num_classes: int,
    baseline_checkpoint: Optional[str] = None,
    device: str = "cuda",
    freeze_backbone: bool = False,
    concept_names: Optional[list] = None,
) -> ConceptBottleneckModel:
    """Tạo CBM, tuỳ chọn load trọng số backbone từ baseline checkpoint.

    Nếu ``baseline_checkpoint`` được cung cấp, chỉ load trọng số phần
    backbone (bỏ qua classifier head của baseline vì CBM có concept layer
    và label predictor riêng).
    """
    model = ConceptBottleneckModel(
        architecture=architecture,
        n_concepts=n_concepts,
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=freeze_backbone,
        concept_names=concept_names,
    )

    if baseline_checkpoint:
        # Load baseline checkpoint và chỉ lấy trọng số backbone
        checkpoint = torch.load(
            baseline_checkpoint,
            map_location=device,
            weights_only=False,
        )
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        # Lọc chỉ backbone weights, bỏ head/classifier
        backbone_state = {}
        for key, value in state_dict.items():
            # CBM backbone keys bắt đầu bằng "backbone."
            if key.startswith("backbone."):
                backbone_state[key] = value

        if backbone_state:
            missing, unexpected = model.load_state_dict(
                backbone_state, strict=False
            )
            # missing sẽ gồm concept_layer + label_predictor (expected)
            # unexpected nên rỗng nếu key mapping đúng

    return model
