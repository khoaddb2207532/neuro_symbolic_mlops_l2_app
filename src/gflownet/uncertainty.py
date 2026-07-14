"""Tính trọng số mẫu (sample_weight) theo độ khó/không chắc chắn của CNN.

MỤC ĐÍCH: chuyển `coverage` trong RuleSetReward từ "tỉ lệ mẫu được phủ"
thành "tỉ lệ TRỌNG SỐ LỖI/UNCERTAINTY được phủ" — luật phủ đúng mẫu mà CNN
đang yếu (dự đoán sai hoặc kém tự tin) sẽ đóng góp coverage cao hơn luật chỉ
phủ mẫu CNN vốn đã đúng/tự tin.

BIẾN THỂ D (hybrid, khuyến nghị dùng): kết hợp error nhị phân + entropy
chuẩn hóa, để tránh việc chỉ dựa vào số mẫu sai (thường rất ít nếu CNN đã
fine-tune tốt) khiến trọng số có phương sai cao / dễ nhiễu:

    u = u_error + lam * (1 - u_error) * u_entropy_normalized

  - Mẫu CNN dự đoán SAI: u ≈ 1 (trọng số đầy đủ, bất kể entropy).
  - Mẫu CNN dự đoán ĐÚNG nhưng kém tự tin (entropy cao): u > 0 một phần,
    theo hệ số `lam` — vẫn được ưu tiên hơn mẫu đúng-và-tự-tin, nhưng không
    bằng mẫu sai hẳn.
  - Mẫu CNN dự đoán ĐÚNG và tự tin (entropy thấp): u ≈ 0.

BẮT BUỘC: tensor `u` trả về phải cùng THỨ TỰ HÀNG với `cover`/`correct` đã
dùng để khởi tạo RuleSetReward (tức cùng tập validation, cùng permutation
mẫu, từ đúng một lần gọi RuleValidator.validate_and_build_tensors() ở
ngoài) — nếu không sẽ trỏ nhầm trọng số sang mẫu khác.

NGUỒN DỮ LIỆU: nên dùng tập val (đã dùng để build cover/correct), KHÔNG
dùng tập test — xem thảo luận trong hội thoại về vì sao val "đủ tách biệt"
với RF (RF chỉ fit trên train) và vì sao test phải giữ nguyên vẹn cho đánh
giá cuối cùng, không được dùng để tính bất kỳ thành phần nào của reward.
"""
from typing import Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_prediction_stats(
    cnn_model: torch.nn.Module,
    features_or_inputs: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
    batch_size: int = 256,
    forward_is_logits: bool = True,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Chạy CNN forward qua tập dữ liệu (thường là val), trả về:
      - pred: (n,) long — nhãn dự đoán (argmax)
      - entropy_norm: (n,) float trong [0,1] — entropy chuẩn hóa của softmax

    `features_or_inputs`: input đưa thẳng vào `cnn_model` (ảnh gốc hoặc
    feature map trung gian, tùy cách `cnn_model` được định nghĩa). Chạy
    theo batch để tránh tràn bộ nhớ với tập val lớn.

    `forward_is_logits`: True nếu `cnn_model(x)` trả về logits (chưa qua
    softmax) — trường hợp phổ biến nhất. Đặt False nếu model đã trả về
    xác suất (softmax output) sẵn.
    """
    cnn_model.eval()
    device = next(cnn_model.parameters()).device

    n = features_or_inputs.shape[0]
    preds = torch.zeros(n, dtype=torch.long)
    entropy_norm = torch.zeros(n, dtype=torch.float)
    log_c = torch.log(torch.tensor(float(num_classes)))

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = features_or_inputs[start:end].to(device)

        out = cnn_model(batch)
        probs = F.softmax(out, dim=-1) if forward_is_logits else out

        preds[start:end] = probs.argmax(dim=-1).cpu()

        ent = -(probs * (probs.clamp(min=1e-12)).log()).sum(dim=-1)
        entropy_norm[start:end] = (ent / log_c).cpu()

    return preds, entropy_norm.clamp(0.0, 1.0)


def compute_prediction_stats_from_logits(
    logits: torch.Tensor,
    num_classes: int,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Giống `compute_prediction_stats`, nhưng đọc thẳng từ LOGITS ĐÃ LƯU SẴN
    (vd `val_logits.pt` do `extract_and_save_features` ghi ra ở stage2) —
    KHÔNG forward lại CNN. Dùng khi pipeline đã lưu logits cùng lúc với
    features, tránh phải nạp checkpoint CNN + forward lại ở stage4 (rủi ro
    lệch checkpoint nếu chọn nhầm phiên bản model).

    logits: (n, num_classes) — CÙNG THỨ TỰ HÀNG với features/labels đã lưu.

    Trả về: pred (n,) long, entropy_norm (n,) float trong [0,1].
    """
    probs = F.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)

    log_c = torch.log(torch.tensor(float(num_classes)))
    ent = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
    entropy_norm = (ent / log_c).clamp(0.0, 1.0)

    return pred.cpu(), entropy_norm.cpu()


def compute_sample_weight(
    pred: torch.Tensor,
    entropy_norm: torch.Tensor,
    y_true: torch.Tensor,
    lam: float = 0.3,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    """Biến thể D — trọng số hybrid error + entropy.

    pred:          (n,) long — nhãn CNN dự đoán (từ compute_prediction_stats)
    entropy_norm:  (n,) float trong [0,1] — entropy chuẩn hóa (từ compute_prediction_stats)
    y_true:        (n,) long — nhãn thật
    lam:           hệ số trộn entropy cho mẫu dự đoán ĐÚNG (mặc định 0.3,
                   xem docstring module để hiểu ý nghĩa)
    clip_max:      nếu khác None, giới hạn trên của u (ví dụ 1.0 hoặc thấp
                   hơn) — cân nhắc dùng nếu quan sát thấy phân phối
                   GFlowNet co cụm quá mạnh vào nhóm mẫu cực khó sau khi
                   thêm sample_weight (xem entropy_norm/unique_ratio khi
                   validate GFlowNet).

    Trả về: u (n,) float, KHÔNG chuẩn hóa tổng — RuleSetReward tự chia cho
    tổng trọng số khi tính coverage weighted.
    """
    assert pred.shape == entropy_norm.shape == y_true.shape, (
        "pred, entropy_norm, y_true phải cùng shape và cùng thứ tự hàng "
        "với cover/correct dùng để khởi tạo RuleSetReward."
    )

    u_error = (pred != y_true).float()
    u = u_error + lam * (1.0 - u_error) * entropy_norm

    if clip_max is not None:
        u = u.clamp(max=clip_max)

    return u


def compute_sample_weight_end_to_end(
    cnn_model: torch.nn.Module,
    features_or_inputs: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
    lam: float = 0.3,
    batch_size: int = 256,
    forward_is_logits: bool = True,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    """Tiện ích gộp: forward CNN + tính u trong một lệnh gọi.

    Dùng ở stage4 (nơi gọi RuleExtractionPipeline.run()), ngay trước khi
    build RuleSetReward, ví dụ:

        u = compute_sample_weight_end_to_end(
            cnn_model=cnn, features_or_inputs=val_inputs, y_true=val_labels,
            num_classes=num_classes,
        )
        pipeline.run(..., sample_weight=u)

    `val_inputs`/`val_labels` PHẢI là đúng tập val (cùng thứ tự mẫu) đã
    dùng để build `cover`/`correct` qua RuleValidator.
    """
    pred, entropy_norm = compute_prediction_stats(
        cnn_model=cnn_model,
        features_or_inputs=features_or_inputs,
        y_true=y_true,
        num_classes=num_classes,
        batch_size=batch_size,
        forward_is_logits=forward_is_logits,
    )
    y_true_cpu = y_true.cpu() if y_true.is_cuda else y_true
    return compute_sample_weight(
        pred=pred, entropy_norm=entropy_norm, y_true=y_true_cpu,
        lam=lam, clip_max=clip_max,
    )