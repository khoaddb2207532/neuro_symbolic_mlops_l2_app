"""Biến thể Stage 4+5: so sánh 3 thuật toán chọn luật (random, top-k-confidence,
greedy-coverage) thay cho GFlowNet, rồi fine-tune CNN với rule-penalty từ tập
luật của TỪNG thuật toán, và tổng hợp bảng so sánh metric cuối cùng.

GIẢ ĐỊNH: stage1 (train_baseline), stage2 (extract_features), stage3
(extract_rules) ĐÃ CHẠY XONG — script này chỉ đọc lại artefact của 3 stage đó
(outputs/01_baseline, outputs/02_features, outputs/03_rules), KHÔNG train lại
CNN baseline, KHÔNG trích luật lại từ RF.

Mỗi thuật toán ghi output vào thư mục RIÊNG (không ghi đè lẫn nhau, không đụng
tới outputs/04_filtered_rules và outputs/05_rules_model của GFlowNet):
  - outputs/04_filtered_rules_<method>/{valid_rules.pkl, selected_rules.pkl, selected_rules.xlsx}
  - outputs/05_rules_model_<method>/{model checkpoint, training curves, test report}

Cuối cùng xuất 1 bảng so sánh: outputs/rule_selection_finetune_comparison.csv
(method, n_rules_selected, test_accuracy, test_f1_macro, save_dir).

LƯU Ý VỀ `random_selection`: đây là phương pháp NGẪU NHIÊN, script này chỉ
chạy 1 lần với 1 seed cố định (--random_seed) vì mỗi lần chạy tốn 1 lần
fine-tune CNN đầy đủ (đắt, khác với việc chỉ đo proxy metric). Nếu cần mean/std
qua nhiều seed cho `random_selection` (đúng chuẩn thống kê hơn), gọi lại script
này nhiều lần với --methods random --random_seed <khác nhau> --run_tag <khác nhau>
rồi tự gộp các dòng trong CSV.
"""
import argparse
import csv
import os
import pickle
from typing import Callable, Dict, List, Tuple

import torch
from sklearn.metrics import f1_score

from src.data.dataset import create_dataloaders, NeuroSymbolicDataset
from src.evaluation.evaluate import evaluate_model_performance, plot_training_history
from src.models.cnn import CNNBaseline
from src.rules.io import save_rules_excel
from src.rules.rule_types import Rule, RuleSet
from src.rules.validator import RuleValidator
from src.training.trainer import train_model
from src.utils.checkpoint import load_model_weights
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

SelectFn = Callable[[List[Rule], torch.Tensor, torch.Tensor, torch.Tensor, int], List[int]]


# --------------------------------------------------------------------------
# 3 thuật toán chọn luật — cùng chữ ký (valid_rules, cover, correct, rule_len,
# max_rules) -> List[int], để dùng thay thế lẫn nhau trong select_rules().
# --------------------------------------------------------------------------
def random_selection(
    valid_rules: List[Rule],
    cover: torch.Tensor,
    correct: torch.Tensor,
    rule_len: torch.Tensor,
    max_rules: int,
    seed: int = 0,
) -> List[int]:
    """Chon ngau nhien max_rules chi so luat. Dung Generator CUC BO (khong
    dung global RNG cua torch) de: (1) tai lap doc lap voi thu tu goi ham
    khac trong script, (2) khong lam lech seed dung cho set_seed() cua qua
    trinh train CNN phia sau."""
    n_rules = cover.shape[0]
    k = min(max_rules, n_rules)
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n_rules, generator=g)
    return perm[:k].tolist()


def topk_confidence(
    valid_rules: List[Rule],
    cover: torch.Tensor,
    correct: torch.Tensor,
    rule_len: torch.Tensor,
    max_rules: int,
) -> List[int]:
    """Sap xep giam dan theo confidence cua tung luat, lay top max_rules.
    Tat dinh, khong can seed."""
    n_rules = len(valid_rules)
    k = min(max_rules, n_rules)
    order = sorted(range(n_rules), key=lambda i: valid_rules[i].confidence, reverse=True)
    return order[:k]


def greedy_coverage(
    valid_rules: List[Rule],
    cover: torch.Tensor,
    correct: torch.Tensor,
    rule_len: torch.Tensor,
    max_rules: int,
) -> List[int]:
    """Greedy submodular don gian: lap `max_rules` lan, moi lan chon luat
    lam tang so mau MOI duoc phu (chua tung duoc phu boi cac luat da chon)
    nhieu nhat. Dung lai neu khong con luat nao tang duoc coverage.
    Vector hoa tren GPU/CPU bang tensor `cover` (n_rules, n_val) bool,
    khong dung vong lap python long qua tung luat trong moi vong (chi
    vong lap ngoai qua `max_rules` buoc, moi buoc 1 phep tinh tensor)."""
    cover_bool = cover.bool()
    n_rules, n_val = cover_bool.shape
    device = cover_bool.device
    k = min(max_rules, n_rules)

    covered = torch.zeros(n_val, dtype=torch.bool, device=device)
    selected_mask = torch.zeros(n_rules, dtype=torch.bool, device=device)
    selected: List[int] = []

    for _ in range(k):
        gains = (cover_bool & ~covered.unsqueeze(0)).sum(dim=1).float()
        gains[selected_mask] = -1.0
        best_idx = int(torch.argmax(gains).item())
        if gains[best_idx].item() <= 0:
            break
        selected.append(best_idx)
        selected_mask[best_idx] = True
        covered |= cover_bool[best_idx]

    return selected


# --------------------------------------------------------------------------
# Tương đương stage4, nhưng tổng quát hoá cho bất kỳ select_fn nào.
# --------------------------------------------------------------------------
def select_rules(
    method_name: str,
    select_fn: SelectFn,
    params: dict,
    device: str,
    output_dir: str,
) -> List[Rule]:
    rules_dir = os.path.join(params["output_dir"], "03_rules")
    features_dir = os.path.join(params["output_dir"], "02_features")

    with open(os.path.join(rules_dir, "raw_rules.pkl"), "rb") as f:
        raw_rules = pickle.load(f)

    val_features = torch.load(f"{features_dir}/val_features.pt").to(device)
    val_labels = torch.load(f"{features_dir}/val_labels.pt").to(device)

    # Dùng LẠI ĐÚNG validator/tham số như stage4-GFlowNet, để đảm bảo tập luật
    # HỢP LỆ đầu vào GIỐNG HỆT nhau giữa các phương pháp — khác biệt DUY NHẤT
    # phải là thuật toán chọn luật, không phải do khác tập luật ứng viên.
    validator = RuleValidator(
        min_supp=params["rules"]["min_support"],
        min_conf=params["rules"]["min_confidence"],
    )
    valid_rule_set, cover, correct, rule_len = validator.validate_and_build_tensors(
        raw_rules, val_features, val_labels, store_device=device
    )
    with open(os.path.join(output_dir, "valid_rules.pkl"), "wb") as f:
        pickle.dump(valid_rule_set, f)

    valid_rules = list(valid_rule_set.rules)
    max_rules = params["gflownet"]["max_rules"]
    selected_indices = select_fn(valid_rules, cover, correct, rule_len, max_rules)
    selected_rules = valid_rule_set.filter_rules(selected_indices).rules  # List[Rule]

    with open(os.path.join(output_dir, "selected_rules.pkl"), "wb") as f:
        pickle.dump(selected_rules, f)
    save_rules_excel(selected_rules, os.path.join(output_dir, "selected_rules.xlsx"))

    logger.info(
        "[%s] Đã chọn %d/%d luật hợp lệ (max_rules=%d).",
        method_name, len(selected_rules), len(valid_rules), max_rules,
    )
    return selected_rules


# --------------------------------------------------------------------------
# Tương đương stage5 (GIỮ NGUYÊN logic fine-tune gốc), tổng quát hoá theo
# method_name/save_dir, và tính thêm F1-macro để phục vụ bảng so sánh
# (evaluate_model_performance gốc chỉ trả về accuracy).
# --------------------------------------------------------------------------
def train_and_evaluate(
    method_name: str,
    selected_rules: List[Rule],
    params: dict,
    device: str,
    baseline_ckpt: str,
    dataloaders_tuple: Tuple,
    class_names: List[str],
    save_dir: str,
) -> Dict:
    _, train_loader, val_loader, test_loader = dataloaders_tuple
    rule_set = RuleSet(rules=selected_rules)
    os.makedirs(save_dir, exist_ok=True)

    if len(rule_set) == 0:
        logger.warning(
            "[%s] rule_set RỖNG — rule-penalty sẽ luôn bằng 0, kết quả sẽ ~ "
            "giống hệt baseline không có luật.", method_name,
        )

    model = CNNBaseline(num_classes=params["num_classes"], freeze_stage="last_block")
    # Nạp lại đúng trọng số baseline (stage1) cho CẢ 3 phương pháp — đảm bảo
    # chênh lệch cuối cùng CHỈ đến từ cách chọn luật, không phải từ 3 baseline
    # xuất phát khác nhau.
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
        "scheduler_factor": 0.1,
        "scheduler_patience": 3,
        "dvclive_path": os.path.join(save_dir, f"dvclive_rule_regularized_{method_name}"),
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

    acc = evaluate_model_performance(
        model, test_loader, device, class_names,
        title=f"Rule-Regularized CNN Performance ({method_name})",
        output_dir=save_dir,
    )
    plot_training_history(history, save_dir=save_dir, title_suffix=f"Rule-Regularized CNN ({method_name})")

    # evaluate_model_performance chỉ trả về accuracy — tính thêm F1-macro
    # riêng (1 lượt forward bổ sung trên test set) để có đủ cột cho bảng so
    # sánh, không sửa hàm gốc trong src/evaluation/evaluate.py.
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    logger.info(
        "[%s] test_accuracy=%.4f test_f1_macro=%.4f (n_rules=%d)",
        method_name, acc, f1_macro, len(selected_rules),
    )
    return {
        "method": method_name,
        "n_rules_selected": len(selected_rules),
        "test_accuracy": acc,
        "test_f1_macro": f1_macro,
        "save_dir": save_dir,
    }


def main(params_path: str, methods: List[str], random_seed: int) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Kiểm tra điều kiện tiên quyết: stage1-3 phải đã chạy xong ---
    baseline_ckpt = os.path.join(params["output_dir"], "01_baseline", "baseline_best.pth")
    features_dir = os.path.join(params["output_dir"], "02_features")
    raw_rules_path = os.path.join(params["output_dir"], "03_rules", "raw_rules.pkl")
    required_paths = [
        baseline_ckpt, raw_rules_path,
        os.path.join(features_dir, "val_features.pt"),
        os.path.join(features_dir, "val_labels.pt"),
    ]
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Thiếu artefact bắt buộc từ stage1/2/3, script này KHÔNG tự train lại:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\nChạy xong stage1 (train_baseline), stage2 (extract_features), "
              "stage3 (extract_rules) trước."
        )

    select_fns: Dict[str, SelectFn] = {
        "random": lambda vr, cov, cor, rl, mr: random_selection(vr, cov, cor, rl, mr, seed=random_seed),
        "topk_confidence": topk_confidence,
        "greedy_coverage": greedy_coverage,
    }
    unknown = [m for m in methods if m not in select_fns]
    if unknown:
        raise ValueError(f"Không nhận diện được method(s) {unknown}. Chọn trong {list(select_fns)}.")

    dataloaders_tuple = create_dataloaders(
        params["data_dir"], batch_size=params["batch_size"], num_workers=params["num_workers"], seed=params["seed"]
    )
    class_names = [
        n for n, _ in sorted(NeuroSymbolicDataset(params["data_dir"], "test").class_to_idx.items(), key=lambda x: x[1])
    ]

    comparison_rows = []
    for method_name in methods:
        logger.info("=== Phương pháp: %s ===", method_name)
        filtered_dir = os.path.join(params["output_dir"], f"04_filtered_rules_{method_name}")
        os.makedirs(filtered_dir, exist_ok=True)
        selected_rules = select_rules(method_name, select_fns[method_name], params, device, filtered_dir)

        save_dir = os.path.join(params["output_dir"], f"05_rules_model_{method_name}")
        result = train_and_evaluate(
            method_name, selected_rules, params, device, baseline_ckpt,
            dataloaders_tuple, class_names, save_dir,
        )
        comparison_rows.append(result)

    comparison_path = os.path.join(params["output_dir"], "rule_selection_finetune_comparison.csv")
    with open(comparison_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "n_rules_selected", "test_accuracy", "test_f1_macro", "save_dir"]
        )
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)

    logger.info("So sánh hoàn tất. Bảng tổng hợp lưu tại: %s", comparison_path)
    for row in comparison_rows:
        logger.info("  %-16s acc=%.4f f1_macro=%.4f n_rules=%d",
                    row["method"], row["test_accuracy"], row["test_f1_macro"], row["n_rules_selected"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument(
        "--methods", nargs="+", default=["random", "topk_confidence", "greedy_coverage"],
        choices=["random", "topk_confidence", "greedy_coverage"],
        help="Chạy 1 hoặc nhiều phương pháp (mặc định cả 3).",
    )
    parser.add_argument(
        "--random_seed", type=int, default=0,
        help="Seed riêng cho random_selection (không dùng chung seed toàn cục), "
             "để tái lập độc lập với thứ tự chạy các phương pháp khác.",
    )
    args = parser.parse_args()
    main(args.config, methods=args.methods, random_seed=args.random_seed)