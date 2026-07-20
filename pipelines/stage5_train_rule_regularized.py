"""DVC Stage 5 — Fine-tune lại CNN với rule penalty (vectorized) từ luật đã chọn.

Model bắt đầu từ ``baseline_best.pth``, mở toàn bộ layer trừ BatchNorm và dùng
cùng learning rate end-to-end với baseline. Khác biệt huấn luyện chính là rule
penalty được cộng vào cross-entropy.
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
    with open(os.path.join(filtered_dir, "selected_rules.pkl"), "rb") as f:
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

    train_cfg = {
        "lr": params["learning_rate"],
        "weight_decay": params["weight_decay"],
        "monitor_metric": params.get("monitor_metric", "val_acc"),
        "use_scheduler": True,
        "scheduler_factor": 0.1,
        "scheduler_patience": 3,
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
        initial_temp=params["rule_penalty"]["initial_temp"],
        final_temp=params["rule_penalty"]["final_temp"],
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
