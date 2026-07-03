"""DVC Stage 5 — Fine-tune lại CNN với rule penalty (vectorized) từ luật đã chọn.

Ở stage này model tiếp tục theo cùng chiến lược transfer learning (progressive
unfreezing + differential LR) nhưng khởi động lại từ baseline đã hội tụ
(baseline_best.pth từ stage 1), KHÔNG khởi tạo lại từ ImageNet — vì mục tiêu
bây giờ là tinh chỉnh sâu hơn có ràng buộc luật dựa trên những gì baseline đã
học được, với freeze_schedule "tiến xa hơn" (được set trong params.yaml).
"""
import argparse
import os
import pickle

import torch

from src.data.dataset import create_dataloaders, NeuroSymbolicDataset
from src.evaluation.evaluate import evaluate_model_performance, plot_training_history
from src.models.cnn import CNNBaseline
from src.rules.rule_types import RuleSet
from src.training.trainer import train_model
from src.utils.checkpoint import load_model_weights
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main(params_path: str) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    filtered_dir = os.path.join(params["output_dir"], "04_filtered_rules")
    with open(os.path.join(filtered_dir, "selected_rules_improved.pkl"), "rb") as f:
        selected_rules = pickle.load(f)
    rule_set = RuleSet(rules=selected_rules)

    dataloaders, train_loader, val_loader, test_loader = create_dataloaders(
        params["data_dir"], batch_size=params["batch_size"], num_workers=params["num_workers"], seed=params["seed"]
    )
    class_names = [
        n for n, _ in sorted(NeuroSymbolicDataset(params["data_dir"], "test").class_to_idx.items(), key=lambda x: x[1])
    ]

    save_dir = os.path.join(params["output_dir"], "05_rules_model")
    os.makedirs(save_dir, exist_ok=True)

    model = CNNBaseline(num_classes=params["num_classes"], freeze_stage="last_block")

    # Nạp trọng số baseline đã hội tụ (stage 1) thay vì tiếp tục từ ImageNet.
    # required=True: đây là điều kiện tiên quyết bắt buộc của stage 5 — nếu
    # chưa chạy stage 1, dừng ngay với lỗi rõ ràng thay vì âm thầm train từ
    # ImageNet (dễ gây nhầm lẫn: kết quả "rule-regularized" sẽ không thật sự
    # kế thừa từ baseline như tên gọi/thiết kế của pipeline).
    baseline_ckpt = os.path.join(params["output_dir"], "01_baseline", "baseline_best.pth")
    load_model_weights(model, baseline_ckpt, device, required=True)

    freeze_schedule = {int(k): v for k, v in params["transfer_learning"]["freeze_schedule_stage2"].items()}
    train_cfg = {
        "lr_backbone": params["transfer_learning"]["lr_backbone"],
        "lr_head": params["transfer_learning"]["lr_head"],
        "weight_decay": params["weight_decay"],
        "freeze_bn": params["transfer_learning"]["freeze_bn"],
        "freeze_schedule": freeze_schedule,
        "monitor_metric": params.get("monitor_metric", "val_acc"),
        "use_scheduler": True,
        "dvclive_path": os.path.join(save_dir, "dvclive_rule_regularized"),
        "save_dir": save_dir,
    }

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        rule_set=rule_set,
        num_epochs=params["num_epochs"],
        patience=params["patience"],
        device=device,
        penalty_weight=params["rule_penalty"]["weight"],
        use_confidence=True,
        smoothing=params["rule_penalty"]["smoothing"],
        num_classes=params["num_classes"],
        config=train_cfg,
    )

    evaluate_model_performance(model, test_loader, device, class_names, title="Rule-Regularized CNN Performance", output_dir=save_dir)
    plot_training_history(history, save_dir=save_dir, title_suffix="Rule-Regularized CNN")
    logger.info("Stage 5 hoàn thành.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
