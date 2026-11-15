"""Sweep ``cbm.concept_weight`` (λ) trong khoảng 0.5 – 5 và tổng hợp kết quả.

Không phải pipeline stage chính thức: mỗi giá trị λ chạy lại ``stage1b_train_cbm`` với
thư mục đầu ra riêng, rồi ghi một dòng vào CSV. Chọn λ theo CSV rồi cập nhật
``params.yaml`` (``cbm.concept_weight``).

Chạy:
    python -m experiments.sweep_concept_weight --config params.yaml \\
        --weights 0.5 1 2 5 --out-dir outputs/cbm_sweep

Mỗi lần chạy tốn gần bằng một lần train CBM đầy đủ; bắt đầu với ít epoch
(sửa ``num_epochs`` trong bản sao config) nếu chỉ cần xếp hạng sơ bộ.
"""
import argparse
import copy
import csv
import json
import os
from typing import Dict, List, Sequence

import yaml

from pipelines import stage1b_train_cbm
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_WEIGHTS = (0.5, 1.0, 2.0, 5.0)


def _weight_tag(weight: float) -> str:
    return f"cw_{weight:g}".replace(".", "p")


def _summarize_history(history: Dict[str, List[float]]) -> Dict[str, float]:
    """Chọn epoch tốt nhất theo val_loss (cùng tiêu chí monitor mặc định) và báo cáo các chỉ số của epoch đó."""
    val_loss = history["val_loss"]
    best_index = min(range(len(val_loss)), key=lambda i: val_loss[i])
    return {
        "epochs_run": float(len(val_loss)),
        "best_epoch": float(best_index + 1),
        "best_val_loss": float(val_loss[best_index]),
        "val_acc_at_best_loss": float(history["val_acc"][best_index]),
        "val_concept_loss_at_best_loss": float(history["val_concept_loss"][best_index]),
        "max_val_acc": float(max(history["val_acc"])),
    }


def run_sweep(config_path: str, weights: Sequence[float], out_dir: str) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for weight in weights:
        tag = _weight_tag(weight)
        run_dir = os.path.join(out_dir, tag)
        os.makedirs(run_dir, exist_ok=True)
        config = copy.deepcopy(base)
        config["cbm"]["concept_weight"] = float(weight)
        config["output_dir"] = run_dir
        run_config_path = os.path.join(run_dir, "params_sweep.yaml")
        with open(run_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

        logger.info("Sweep concept_weight=%g -> %s", weight, run_dir)
        stage1b_train_cbm.main(run_config_path)

        with open(os.path.join(run_dir, "cbm", "history.json"), "r", encoding="utf-8") as f:
            history = json.load(f)
        rows.append({"concept_weight": float(weight), **_summarize_history(history)})

    summary_path = os.path.join(out_dir, "concept_weight_sweep.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Đã ghi kết quả sweep vào %s", summary_path)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--out-dir", default="outputs/cbm_sweep")
    args = parser.parse_args()
    for weight in args.weights:
        if not 0.0 < weight <= 10.0:
            parser.error(f"concept_weight {weight} ngoài khoảng hợp lý (0, 10].")
    run_sweep(args.config, args.weights, args.out_dir)


if __name__ == "__main__":
    main()
