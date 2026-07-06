"""GOI A — Ablation 4 chiến lược Transfer Learning (freeze_schedule).

Đặt file này cạnh `pipelines/` (cùng cấp thư mục gốc repo, nơi có `params.yaml`),
KHÔNG đặt bên trong package `pipelines/` (script này không phải 1 DVC stage,
mà là 1 tool ablation gọi lại logic của stage1 nhiều lần với các biến thể
`transfer_learning.freeze_schedule` khác nhau + nhiều seed).

────────────────────────────────────────────────────────────────────────────
4 BIẾN THỂ (so với `freeze_schedule` gốc trong params.yaml: progressive
head_only -> last_block -> full):

  (a) head_only_forever   : freeze_schedule = {0: "head_only"}
                             — không bao giờ mở backbone.
  (b) diff_lr_no_freeze   : freeze_schedule = {0: "full"}
                             — mở toàn bộ backbone NGAY epoch 0, nhưng vẫn
                               dùng Differential LR (lr_head != lr_backbone,
                               lấy đúng 2 giá trị `lr_head`/`lr_backbone` có
                               sẵn trong params.yaml gốc).
  (c) full_finetune       : freeze_schedule = {0: "full"}
                             — mở toàn bộ backbone NGAY epoch 0, và KHÔNG
                               dùng differential LR: gán lr_backbone = lr_head
                               (cùng 1 giá trị) để 2 param-group trong
                               optimizer có LR bằng nhau, tương đương "1 LR
                               duy nhất cho toàn mạng".
  (d) progressive         : dùng đúng freeze_schedule gốc trong params.yaml
                             (head_only -> last_block -> full).

────────────────────────────────────────────────────────────────────────────
GIẢ ĐỊNH (ASSUMPTIONS) — vì không có quyền sửa các file gốc đã đính kèm và
không được bịa tham số ngoài params.yaml:

1. `set_freeze_stage` của `CNNBaseline` chỉ nhận 3 giá trị hợp lệ:
   "head_only" | "last_block" | "full" (xem src/models/cnn.py). Vì vậy biến
   thể (b) và (c) đều dùng freeze_schedule={0: "full"} — không có state
   "mở hoàn toàn nhưng không qua last_block" nào khác để biểu diễn "bỏ qua
   head_only" theo nghĩa đen; "full" ngay từ epoch 0 là cách duy nhất model
   hỗ trợ sẵn để "không freeze gì từ đầu". Khác biệt thực sự giữa (b) và (c)
   nằm ở tầng optimizer: (b) giữ differential LR như thiết kế gốc, (c) triệt
   tiêu differential LR bằng cách set lr_backbone = lr_head.
2. `train_model(...)` (src/training/trainer.py) đã tự load lại
   `best_weights` (theo `monitor_metric` trong params.yaml, mặc định
   "val_loss") vào `model` trước khi return — nên model trả về CHÍNH LÀ
   checkpoint tốt nhất của epoch đó. `test_acc` / `test_f1_macro` trong CSV
   được tính trên model này (tương đương "best" checkpoint), không phải
   epoch cuối cùng.
3. `history` dict trả về từ `train_model` có đúng các key:
   "train_loss", "train_ce", "train_penalty", "train_acc", "val_loss",
   "val_acc" (mỗi key là list 1 giá trị / epoch, xem trainer.py dòng
   history = {...}). `val_acc_best_epoch` trong CSV = giá trị
   history["val_acc"][idx_best], với idx_best = index đạt best theo
   MONITOR_MODES[monitor_metric] (import trực tiếp từ
   src.training.trainer, không định nghĩa lại logic so sánh best).
4. `evaluate_model_performance(...)` (src/evaluation/evaluate.py) chỉ trả về
   accuracy (float) và ghi confusion-matrix/classification-report ra đĩa —
   KHÔNG trả về F1-macro và cũng không trả về (preds, labels) thô. Vì vậy
   file này định nghĩa thêm 1 hàm nhỏ `_predict_test_set()` (không sửa
   evaluate.py) chỉ để lấy list (preds, labels) trên test set, rồi tự tính
   F1-macro bằng sklearn.metrics.f1_score. `evaluate_model_performance` vẫn
   được gọi y nguyên (để tận dụng việc ghi confusion matrix + classification
   report có sẵn) — accuracy trả về từ đó phải khớp với accuracy tính lại
   trong `_predict_test_set()` (script assert việc này để phát hiện sai lệch
   sớm nếu format model/dataloader thay đổi trong tương lai).
5. `NeuroSymbolicDataset(...).class_to_idx` (src/data/dataset.py) ổn định
   giữa các lần khởi tạo (do `_find_classes()` dùng `sorted()`) nên
   `class_names` suy ra từ 1 lần khởi tạo dataset "test" là đáng tin cậy,
   giống cách `pipelines/stage1_train_baseline.py` đang làm.
6. Model forward luôn trả về tuple `(logits, features)` — đúng interface
   của `CNNBaseline.forward` (src/models/cnn.py) — không có nhánh nào khác.
7. Mỗi (variant, seed) chạy hoàn toàn độc lập: DataLoader/model/optimizer
   mới hoàn toàn (gọi lại `create_dataloaders` mỗi lần) để đảm bảo seed
   ảnh hưởng đúng tới cả thứ tự shuffle train lẫn khởi tạo trọng số
   (qua `set_seed`), không tái sử dụng state giữa các lần chạy.
8. `params.yaml` sinh ra cho mỗi biến thể là 1 file YAML độc lập (copy đầy đủ
   từ params.yaml gốc, chỉ đổi `transfer_learning.freeze_schedule` — và với
   biến thể (c) là `transfer_learning.lr_backbone`) — file gốc mà người dùng
   đưa vào (`--config`) KHÔNG bị ghi đè.

────────────────────────────────────────────────────────────────────────────
CÁCH CHẠY:

    python run_ablation_freeze.py --config params.yaml \
        --seeds 42 43 44 \
        --output_dir outputs/ablation_freeze

Tham số:
  --config       đường dẫn params.yaml gốc (mặc định "params.yaml")
  --seeds        danh sách seed (mặc định 42 43 44)
  --output_dir   thư mục gốc chứa toàn bộ output của ablation
                 (mặc định "outputs/ablation_freeze")
  --variants     (tùy chọn) lọc bớt biến thể muốn chạy, vd
                 `--variants head_only_forever progressive`
                 (mặc định: chạy cả 4)

CẤU TRÚC OUTPUT sau khi chạy xong (với --output_dir outputs/ablation_freeze):

    outputs/ablation_freeze/
    ├── configs/
    │   ├── params_head_only_forever.yaml
    │   ├── params_diff_lr_no_freeze.yaml
    │   ├── params_full_finetune.yaml
    │   └── params_progressive.yaml
    ├── head_only_forever/
    │   ├── seed_42/
    │   │   ├── baseline_best.pth, final_model_weights.pth   (checkpoint)
    │   │   ├── dvclive/                                     (log DVCLive)
    │   │   ├── *_confusion_matrix.png, *_classification_report.txt
    │   │   └── training_metrics_*.png                       (curve loss/acc)
    │   ├── seed_43/ ...
    │   └── seed_44/ ...
    ├── diff_lr_no_freeze/seed_{42,43,44}/...
    ├── full_finetune/seed_{42,43,44}/...
    ├── progressive/seed_{42,43,44}/...
    └── ablation_results.csv     <-- FILE KẾT QUẢ TỔNG HỢP DUY NHẤT
        cột: variant, seed, test_acc, test_f1_macro, val_acc_best_epoch

Đọc kết quả: mở `ablation_results.csv` bằng pandas/Excel, group-by `variant`
rồi lấy mean/std của `test_acc` và `test_f1_macro` qua 3 seed, vd:

    import pandas as pd
    df = pd.read_csv("outputs/ablation_freeze/ablation_results.csv")
    print(df.groupby("variant")[["test_acc", "test_f1_macro"]].agg(["mean", "std"]))
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
from typing import Dict, List, Tuple

import torch
import yaml
from sklearn.metrics import f1_score

from src.data.dataset import NeuroSymbolicDataset, create_dataloaders
from src.evaluation.evaluate import evaluate_model_performance, plot_training_history
from src.models.cnn import CNNBaseline
from src.training.trainer import MONITOR_MODES, train_model
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Định nghĩa 4 biến thể freeze_schedule (xem giải thích đầy đủ ở docstring).
# Mỗi entry: (freeze_schedule, "differential_lr" | "single_lr")
#   - "differential_lr": giữ nguyên lr_head/lr_backbone từ params.yaml gốc.
#   - "single_lr": ghi đè lr_backbone = lr_head trong bản copy params.yaml
#     của biến thể đó (chỉ áp dụng cho "full_finetune").
# ---------------------------------------------------------------------------
VARIANTS: Dict[str, Dict] = {
    "head_only_forever": {
        "freeze_schedule": {0: "head_only"},
        "lr_mode": "differential_lr",
    },
    "diff_lr_no_freeze": {
        "freeze_schedule": {0: "full"},
        "lr_mode": "differential_lr",
    },
    "full_finetune": {
        "freeze_schedule": {0: "full"},
        "lr_mode": "single_lr",
    },
    "progressive": {
        "freeze_schedule": None,  # None => giữ nguyên freeze_schedule gốc trong params.yaml
        "lr_mode": "differential_lr",
    },
}


def _make_variant_params(base_params: Dict, variant_name: str, variant_cfg: Dict) -> Dict:
    """Tạo 1 bản copy sâu (deep copy) của params gốc, áp freeze_schedule +
    lr_mode của biến thể. KHÔNG sửa `base_params` (giữ nguyên file gốc)."""
    params = copy.deepcopy(base_params)
    tl = params["transfer_learning"]

    if variant_cfg["freeze_schedule"] is not None:
        tl["freeze_schedule"] = dict(variant_cfg["freeze_schedule"])
    # else: giữ nguyên freeze_schedule gốc (biến thể "progressive")

    if variant_cfg["lr_mode"] == "single_lr":
        # "full fine-tune" không differential LR: cùng 1 giá trị cho cả
        # head lẫn backbone (xem assumption 1 ở docstring đầu file).
        tl["lr_backbone"] = tl["lr_head"]

    return params


def _dump_variant_yaml(params: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, allow_unicode=True, sort_keys=False)


def _predict_test_set(
    model: torch.nn.Module, test_loader, device
) -> Tuple[List[int], List[int]]:
    """Lấy (preds, labels) thô trên test set.

    Hàm mới, KHÔNG có sẵn trong src/evaluation/evaluate.py — cần vì
    `evaluate_model_performance` chỉ trả về accuracy (không trả preds/labels
    nên không tự tính được F1-macro từ nó). Logic forward pass giống hệt
    vòng lặp trong `evaluate_model_performance` (đúng interface
    `model(x) -> (logits, features)` của CNNBaseline).
    """
    model = model.to(device)
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return all_preds, all_labels


def _best_epoch_val_acc(history: Dict, monitor_metric: str) -> float:
    """Lấy val_acc tại epoch "tốt nhất" theo đúng tiêu chí monitor_metric mà
    trainer.py dùng để quyết định checkpoint tốt nhất (xem assumption 3).
    """
    mode = MONITOR_MODES[monitor_metric]
    series = history[monitor_metric]
    if not series:
        return float("nan")
    if mode == "max":
        idx = max(range(len(series)), key=lambda i: series[i])
    else:
        idx = min(range(len(series)), key=lambda i: series[i])
    return history["val_acc"][idx]


def run_one(
    variant_name: str,
    variant_params: Dict,
    seed: int,
    output_root: str,
) -> Dict:
    """Chạy train + evaluate cho đúng 1 (variant, seed). Trả về 1 dict record
    (1 dòng của CSV kết quả)."""
    run_dir = os.path.join(output_root, variant_name, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = variant_params["data_dir"]
    dataloaders, train_loader, val_loader, test_loader = create_dataloaders(
        data_dir,
        batch_size=variant_params["batch_size"],
        num_workers=variant_params["num_workers"],
        seed=seed,
    )
    class_names = [
        n
        for n, _ in sorted(
            NeuroSymbolicDataset(data_dir, "test").class_to_idx.items(), key=lambda x: x[1]
        )
    ]

    tl = variant_params["transfer_learning"]
    freeze_schedule = {int(k): v for k, v in tl["freeze_schedule"].items()}
    initial_stage = freeze_schedule[min(freeze_schedule)]

    model = CNNBaseline(num_classes=variant_params["num_classes"], freeze_stage=initial_stage)

    train_cfg = {
        "lr_backbone": tl["lr_backbone"],
        "lr_head": tl["lr_head"],
        "weight_decay": variant_params["weight_decay"],
        "freeze_bn": tl["freeze_bn"],
        "freeze_schedule": freeze_schedule,
        "monitor_metric": variant_params.get("monitor_metric", "val_acc"),
        "use_scheduler": True,
        "scheduler_factor": 0.1,
        "scheduler_patience": 3,
        "dvclive_path": os.path.join(run_dir, "dvclive"),
        "save_dir": run_dir,
    }

    logger.info("=== Variant '%s' | seed %d | freeze_schedule=%s ===", variant_name, seed, freeze_schedule)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        rule_set=None,
        num_epochs=variant_params["num_epochs"],
        patience=variant_params["patience"],
        device=device,
        penalty_weight=0.0,
        num_classes=variant_params["num_classes"],
        config=train_cfg,
    )

    title = f"{variant_name}_seed{seed}"
    acc_from_eval = evaluate_model_performance(
        model, test_loader, device, class_names, title=title, output_dir=run_dir
    )
    plot_training_history(history, save_dir=run_dir, title_suffix=title)

    preds, labels = _predict_test_set(model, test_loader, device)
    test_acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    test_f1_macro = f1_score(labels, preds, average="macro")

    # Sanity check: accuracy tính lại phải khớp accuracy mà
    # evaluate_model_performance đã tính (xem assumption 4).
    if abs(test_acc - acc_from_eval) > 1e-6:
        logger.warning(
            "Lệch accuracy giữa evaluate_model_performance (%.6f) và tính lại thủ công (%.6f) "
            "cho variant=%s seed=%d — kiểm tra lại test_loader có bị shuffle/không deterministic không.",
            acc_from_eval, test_acc, variant_name, seed,
        )

    val_acc_best_epoch = _best_epoch_val_acc(history, train_cfg["monitor_metric"])

    return {
        "variant": variant_name,
        "seed": seed,
        "test_acc": test_acc,
        "test_f1_macro": test_f1_macro,
        "val_acc_best_epoch": val_acc_best_epoch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation 4 chiến lược transfer learning (freeze_schedule)."
    )
    parser.add_argument("--config", default="params.yaml", help="Đường dẫn params.yaml gốc")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44], help="Danh sách seed (>=3 theo yêu cầu)"
    )
    parser.add_argument(
        "--output_dir", default="outputs/ablation_freeze", help="Thư mục gốc chứa output ablation"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
        help="Lọc bớt biến thể muốn chạy (mặc định: chạy cả 4)",
    )
    args = parser.parse_args()

    base_params = load_params(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Sinh 4 (hoặc ít hơn nếu lọc bằng --variants) bản copy params.yaml,
    #    KHÔNG sửa file gốc `args.config`.
    configs_dir = os.path.join(args.output_dir, "configs")
    variant_params_map: Dict[str, Dict] = {}
    for variant_name in args.variants:
        variant_params = _make_variant_params(base_params, variant_name, VARIANTS[variant_name])
        variant_params_map[variant_name] = variant_params
        _dump_variant_yaml(variant_params, os.path.join(configs_dir, f"params_{variant_name}.yaml"))
        logger.info("Đã sinh config cho variant '%s' tại %s", variant_name, configs_dir)

    # 2) Train + evaluate cho mọi (variant x seed).
    records: List[Dict] = []
    for variant_name in args.variants:
        variant_params = variant_params_map[variant_name]
        for seed in args.seeds:
            record = run_one(variant_name, variant_params, seed, args.output_dir)
            records.append(record)
            logger.info(
                "Xong variant=%s seed=%d | test_acc=%.4f | test_f1_macro=%.4f | val_acc_best_epoch=%.4f",
                record["variant"], record["seed"], record["test_acc"],
                record["test_f1_macro"], record["val_acc_best_epoch"],
            )

    # 3) Gộp kết quả vào 1 file CSV duy nhất.
    csv_path = os.path.join(args.output_dir, "ablation_results.csv")
    fieldnames = ["variant", "seed", "test_acc", "test_f1_macro", "val_acc_best_epoch"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    logger.info("Ablation hoàn thành. Kết quả tổng hợp tại: %s", csv_path)


if __name__ == "__main__":
    main()