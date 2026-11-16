"""Chạy tiếp cho MỘT seed từ CBM đã train ở lần chạy trước (luồng trước khi sửa).

Giả định ``output_dir`` đã chứa kết quả lần chạy trước (``cbm/``, ``04_filtered_rules/``,
``05_rules_model/``, ...). Không chạy lại Baseline/Stage 1b/Stage 2/Stage 3/Stage 4 cũ.

Thứ tự:
  0. Kiểm tra checkpoint CBM đã khôi phục: đúng backbone, đúng số concept.
  1. Stage 4C: GFlowNet xây luật từ image_concept (``04c_concept_rules/``).
  2. Stage 5 với luật mới, lần lượt ``penalty`` | ``head`` | ``both``
     (``05c_concept_<mode>/``).
  3. Bảng so sánh ``concept_rules_comparison.csv``: các mô hình cũ đã có + các mô hình mới.

Chạy:
  python -m pipelines.run_concept_rules_pipeline --config params_seed_42.yaml
"""
import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from pipelines import stage4_concept_rules_gflownet, stage5_train_rule_regularized
from pipelines.run_cbm_neuro_symbolic_pipeline import EXPECTED_MODELS
from src.utils.config import load_params, selected_baseline_architecture
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MODES: tuple[str, ...] = ("penalty", "head", "both")
RULE_SUBDIR = "04c_concept_rules"


def check_restored_cbm(params: dict[str, Any]) -> None:
    """Dừng sớm nếu checkpoint CBM khôi phục không khớp cấu hình hiện tại."""
    cbm_dir = Path(params["output_dir"]) / "cbm"
    ckpt = cbm_dir / "cbm_best.pth"
    if not ckpt.exists():
        ckpt = cbm_dir / "final_model_weights.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Không có checkpoint CBM tại {cbm_dir}. Cần khôi phục outputs của lần chạy trước."
        )
    meta_path = cbm_dir / "cbm_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Thiếu {meta_path}.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    arch = selected_baseline_architecture(params)
    if meta.get("architecture") != arch:
        raise ValueError(
            f"CBM đã train với backbone '{meta.get('architecture')}' nhưng config chọn '{arch}'."
        )
    n_concepts = int(params.get("cbm", {}).get("n_concepts", -1))
    if int(meta.get("n_concepts", -2)) != n_concepts:
        raise ValueError(
            f"CBM có {meta.get('n_concepts')} concept, config có {n_concepts}."
        )
    logger.info("CBM khôi phục hợp lệ: %s, %d concept, từ %s", arch, n_concepts, ckpt)


def _classification_row(model_name: str, save_dir: Path, seed: int) -> dict[str, Any] | None:
    reports = list(save_dir.glob("*classification_report.json"))
    if not reports:
        return None
    data = json.loads(reports[0].read_text(encoding="utf-8"))
    return {
        "Seed": seed,
        "Model": model_name,
        "Accuracy (%)": round(data.get("accuracy", 0.0) * 100, 2),
        "Macro F1 (%)": round(data.get("macro avg", {}).get("f1-score", 0.0) * 100, 2),
        "Weighted F1 (%)": round(data.get("weighted avg", {}).get("f1-score", 0.0) * 100, 2),
    }


def write_comparison(output_dir: Path, seed: int, new_models: Sequence[tuple[str, str]]) -> Path:
    """Gộp các mô hình cũ (đã train) và mô hình mới vào một bảng CSV."""
    import pandas as pd

    rows = []
    for name, subdir in list(EXPECTED_MODELS) + list(new_models):
        row = _classification_row(name, output_dir / subdir, seed)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("Không có classification report nào để so sánh.")
    df = pd.DataFrame(rows)
    out = output_dir / "concept_rules_comparison.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info("Bảng so sánh:\n%s", df.to_string(index=False))
    return out


def main(config_path: str, modes: Sequence[str] = DEFAULT_MODES) -> None:
    params = load_params(config_path)
    seed = int(params["seed"])
    output_dir = Path(params["output_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=== Seed %d | output %s | device %s ===", seed, output_dir, device)

    check_restored_cbm(params)

    logger.info(">>> [1/3] STAGE 4C: GFlowNet xây luật từ image_concept")
    stage4_concept_rules_gflownet.main(config_path)

    new_models: list[tuple[str, str]] = []
    for mode in modes:
        subdir = f"05c_concept_{mode}"
        logger.info(">>> [2/3] STAGE 5 | rules=%s | mode=%s -> %s", RULE_SUBDIR, mode, subdir)
        stage5_train_rule_regularized.main(
            config_path,
            target_override="cbm",
            save_subdir=subdir,
            rules_dir_override=RULE_SUBDIR,
            mode_override=mode,
        )
        new_models.append((f"CBM + concept rules ({mode})", subdir))

    logger.info(">>> [3/3] Bảng so sánh")
    write_comparison(output_dir, seed, new_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=list(DEFAULT_MODES))
    args = parser.parse_args()
    main(args.config, modes=args.modes)
