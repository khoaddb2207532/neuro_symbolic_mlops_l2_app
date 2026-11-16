"""Stage 4C — GFlowNet XÂY DỰNG luật trên image_concept (không chọn từ luật RF có sẵn).

Khác Stage 3/4 (``stage3_extract_rules`` + ``stage4_select_rules_gflownet``):
  - Không cần cây quyết định / RandomForest trên đặc trưng.
  - Luật là conjunction của literal concept (``áo dài ∧ chùa ⇒ khmer``) được sinh bởi GFlowNet.
  - Reward đo trên nhãn concept ground-truth của ``rule_source_split`` (mặc định train).
    Val/test chỉ dùng để BÁO CÁO trong ``concept_rules_report.json``, không ảnh hưởng chọn luật.

Đầu ra (``output_dir/<concept_rules.output_subdir>``, mặc định ``04c_concept_rules``):
  - selected_rules.pkl       RuleSet — cùng định dạng Stage 4 nên Stage 5 đọc được ngay
  - concept_rules.json/.txt  luật đọc được bằng tiếng Việt, kèm precision/recall
  - gflownet_concept_policy.pth, concept_rules_report.json

Chạy:  python -m pipelines.stage4_concept_rules_gflownet --config params.yaml
Sau đó: python -m pipelines.stage5_train_rule_regularized --target cbm --rules-dir 04c_concept_rules
"""
import argparse
import dataclasses
import json
import os
from typing import Any

import torch

from pipelines.stage1b_train_cbm import wrap_with_concepts
from src.data.dataset import NeuroSymbolicDataset, create_dataloaders
from src.gflownet.concept_rule_pipeline import (
    ConceptRuleGFlowNetPipeline,
    ConceptRuleSearchConfig,
    evaluate_rule_set,
)
from src.utils.config import load_params
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _split_matrix(
    wrapped: dict[str, Any], split: str, use_prototype: bool, n_concepts: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(A [N,K] có NaN = ignore, labels [N], số mẫu bị loại vì không có nhãn riêng)."""
    ds = wrapped[split].dataset
    rows: list[list[float]] = []
    labels: list[int] = []
    skipped = 0
    for idx in range(len(ds)):
        vector, used_fallback = ds.target_vector(idx)
        if used_fallback and not use_prototype:
            # Prototype lớp là nhãn SUY RA, không phải nhãn gán cho ảnh. Dùng nó để học
            # luật sẽ khiến luật chỉ học lại chính prototype, nên loại khỏi reward.
            skipped += 1
            vector = [float("nan")] * n_concepts
        rows.append([float(v) for v in vector])
        labels.append(int(ds.base_dataset.samples[idx][1]))
    matrix = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros(0, n_concepts)
    return matrix, torch.tensor(labels, dtype=torch.long), skipped


def _search_config(raw: dict[str, Any]) -> ConceptRuleSearchConfig:
    known = {f.name for f in dataclasses.fields(ConceptRuleSearchConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"concept_rules có khóa không hợp lệ: {sorted(unknown)}")
    return ConceptRuleSearchConfig(**raw)


def main(params_path: str) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_raw: dict[str, Any] = dict(params.get("concept_rules", {}))
    source_split = str(cfg_raw.pop("source_split", "train"))
    use_prototype = bool(cfg_raw.pop("use_class_prototype", False))
    output_subdir = str(cfg_raw.pop("output_subdir", "04c_concept_rules"))
    cfg = _search_config(cfg_raw)
    if cfg.seed is None:
        cfg.seed = int(params["seed"])

    dataloaders, _, _, _ = create_dataloaders(
        params["data_dir"], batch_size=params["batch_size"],
        num_workers=params["num_workers"], seed=params["seed"],
    )
    class_names = [
        n for n, _ in sorted(
            NeuroSymbolicDataset(params["data_dir"], "test").class_to_idx.items(), key=lambda x: x[1]
        )
    ]
    cbm_dir = os.path.join(params["output_dir"], "cbm")
    wrapped, concept_names, n_concepts = wrap_with_concepts(dataloaders, params, class_names, cbm_dir)

    A_src, y_src, skipped = _split_matrix(wrapped, source_split, use_prototype, n_concepts)
    n_labeled = int((~torch.isnan(A_src)).any(dim=1).sum())
    logger.info(
        "Split '%s': %d mẫu, %d có nhãn concept, %d bị loại (dùng prototype lớp).",
        source_split, len(y_src), n_labeled, skipped,
    )
    if n_labeled == 0:
        raise RuntimeError(
            f"Split '{source_split}' không có mẫu nào có nhãn concept riêng — không thể học luật. "
            "Kiểm tra cbm.concept_annotations_path."
        )

    out_dir = os.path.join(params["output_dir"], output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    pipeline = ConceptRuleGFlowNetPipeline(device=device, cfg=cfg)
    result = pipeline.run(
        concept_matrix=A_src,
        labels=y_src,
        num_classes=int(params["num_classes"]),
        concept_names=concept_names,
        class_names=class_names,
        output_dir=out_dir,
    )

    # Báo cáo độc lập trên mọi split — chỉ để đọc kết quả, không tác động tới luật đã chọn.
    report: dict[str, Any] = {
        "source_split": source_split,
        "n_rules": len(result.rules),
        "archive_size": result.archive_size,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        if split not in wrapped:
            continue
        A, y, _ = _split_matrix(wrapped, split, use_prototype=False, n_concepts=n_concepts)
        report["splits"][split] = evaluate_rule_set(
            result.candidates, A, y, int(params["num_classes"])
        )
    with open(os.path.join(out_dir, "concept_rules_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(
        "Stage 4C hoàn thành: %d luật -> %s. Coverage/acc test: %s",
        len(result.rules), out_dir, report["splits"].get("test"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
