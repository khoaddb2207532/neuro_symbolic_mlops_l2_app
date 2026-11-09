"""DVC Stage 1b — Huấn luyện Concept Bottleneck Model (CBM).

Chạy: python -m pipelines.stage1b_train_cbm --config params.yaml

Workflow:
  1. Load dataset + concept annotations (hoặc tạo template class-concept).
  2. Tạo ConceptBottleneckModel từ backbone pretrained.
  3. Huấn luyện với joint loss: task_loss + λ · concept_loss.
  4. Evaluate + lưu checkpoint.

Nếu chưa có file concept annotations, script tự tạo template tại
``outputs/cbm/class_concepts_template.json`` — người dùng cần chỉnh sửa
template này cho phù hợp rồi chạy lại.
"""
import argparse
import json
import os

import torch

from src.data.dataset import create_dataloaders, NeuroSymbolicDataset
from src.data.concepts import (
    ConceptAnnotatedDataset,
    build_class_to_concepts,
    generate_default_class_concepts,
    load_concept_annotations,
)
from src.evaluation.evaluate import evaluate_model_performance, plot_training_history
from src.models.cbm import ConceptBottleneckModel, create_cbm_from_baseline
from src.training.cbm_trainer import train_cbm
from src.utils.config import load_params, selected_baseline_architecture
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _wrap_with_concepts(
    dataloaders: dict,
    params: dict,
    class_names: list,
    save_dir: str,
):
    """Wrap dataloaders gốc thành ConceptAnnotatedDataset.

    Thứ tự ưu tiên:
      1. File concept_annotations_path (per-image annotations) — chính xác nhất.
      2. File class_concept_matrix_path (per-class annotations) — khả thi.
      3. Tự sinh template mẫu → yêu cầu người dùng chỉnh sửa.
    """
    cbm_cfg = params.get("cbm", {})
    n_concepts = cbm_cfg.get("n_concepts", 30)

    annotations_path = cbm_cfg.get("concept_annotations_path")
    matrix_path = cbm_cfg.get("class_concept_matrix_path")

    concept_annotations = None
    class_to_concepts = None
    concept_names = None

    if annotations_path and os.path.exists(annotations_path):
        res = load_concept_annotations(annotations_path, class_names=class_names)
        concept_annotations = res[0]
        concept_names = res[1]
        n_concepts = res[2]
        if len(res) > 3 and res[3] is not None:
            class_to_concepts = res[3]
        logger.info(
            "Loaded per-image concept annotations: %d sample keys, %d concepts.",
            len(concept_annotations), n_concepts,
        )
    elif matrix_path and os.path.exists(matrix_path):
        class_to_concepts, concept_names, n_concepts = build_class_to_concepts(
            matrix_path
        )
        logger.info(
            "Loaded class-to-concept matrix: %d classes, %d concepts.",
            len(class_to_concepts), n_concepts,
        )
    else:
        # Tạo template mẫu
        template_path = os.path.join(save_dir, "class_concepts_template.json")
        class_to_concepts, concept_names = generate_default_class_concepts(
            class_names, n_concepts=n_concepts, output_path=template_path,
        )
        logger.warning(
            "Không tìm thấy file concept annotations. Đã tạo TEMPLATE tại:\n"
            "  %s\n"
            "Hãy chỉnh sửa template này (đổi concept_names & class_concepts) "
            "cho phù hợp với domain, rồi cập nhật đường dẫn trong params.yaml:\n"
            "  cbm.class_concept_matrix_path: '%s'\n"
            "Hiện tại dùng template tự sinh (kết quả sẽ kém).",
            template_path, template_path,
        )

    wrapped = {}
    for split_name, loader in dataloaders.items():
        ds = ConceptAnnotatedDataset(
            base_dataset=loader.dataset,
            concept_annotations=concept_annotations,
            class_to_concepts=class_to_concepts,
            n_concepts=n_concepts,
        )
        wrapped[split_name] = torch.utils.data.DataLoader(
            ds,
            batch_size=loader.batch_size,
            shuffle=(split_name == "train"),
            drop_last=(split_name == "train"),
            num_workers=loader.num_workers,
            pin_memory=True,
        )

    return wrapped, concept_names, n_concepts


def main(params_path: str) -> None:
    params = load_params(params_path)
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cbm_cfg = params.get("cbm", {})
    n_concepts = cbm_cfg.get("n_concepts", 30)
    concept_weight = cbm_cfg.get("concept_weight", 1.0)

    # Load data
    dataloaders, _, _, _ = create_dataloaders(
        params["data_dir"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        seed=params["seed"],
    )
    class_names = [
        n for n, _ in sorted(
            NeuroSymbolicDataset(params["data_dir"], "test").class_to_idx.items(),
            key=lambda x: x[1],
        )
    ]

    save_dir = os.path.join(params["output_dir"], "cbm")
    os.makedirs(save_dir, exist_ok=True)

    # Wrap dataloaders với concept annotations
    wrapped_loaders, concept_names, n_concepts = _wrap_with_concepts(
        dataloaders, params, class_names, save_dir,
    )

    # Tạo CBM model
    architecture = selected_baseline_architecture(params)
    baseline_ckpt = cbm_cfg.get("baseline_checkpoint")
    freeze_backbone = cbm_cfg.get("freeze_backbone", False)

    model = create_cbm_from_baseline(
        architecture=architecture,
        n_concepts=n_concepts,
        num_classes=params["num_classes"],
        baseline_checkpoint=baseline_ckpt,
        device=device,
        freeze_backbone=freeze_backbone,
        concept_names=concept_names,
    )

    logger.info(
        "CBM created: backbone=%s, n_concepts=%d, num_classes=%d, "
        "freeze_backbone=%s, concept_weight=%.2f",
        architecture, n_concepts, params["num_classes"],
        freeze_backbone, concept_weight,
    )

    train_cfg = {
        "lr_backbone": params["transfer_learning"]["lr_backbone"],
        "lr_head": params["transfer_learning"]["lr_head"],
        "weight_decay": params["weight_decay"],
        "monitor_metric": params.get("monitor_metric", "val_loss"),
        "dvclive_path": os.path.join(save_dir, "dvclive_cbm"),
        "save_dir": save_dir,
    }

    model, history = train_cbm(
        model=model,
        train_loader=wrapped_loaders["train"],
        val_loader=wrapped_loaders["val"],
        num_epochs=params["num_epochs"],
        lr=params["transfer_learning"]["lr_head"],
        patience=params["patience"],
        device=device,
        concept_weight=concept_weight,
        smoothing=cbm_cfg.get("smoothing", 0.1),
        num_classes=params["num_classes"],
        config=train_cfg,
    )

    # Evaluate
    # Lưu ý: evaluate_model_performance expect model(images) → (logits, features)
    # CBM forward trả về (logits, concept_probs, raw_features), cần wrapper
    class _CBMEvalWrapper(torch.nn.Module):
        def __init__(self, cbm):
            super().__init__()
            self.cbm = cbm
        def forward(self, x):
            logits, concept_probs, _ = self.cbm(x)
            return logits, concept_probs

    eval_model = _CBMEvalWrapper(model)
    evaluate_model_performance(
        eval_model, dataloaders["test"], device, class_names,
        title="CBM Performance", output_dir=save_dir,
    )
    plot_training_history(history, save_dir=save_dir, title_suffix="CBM")

    # Lưu concept metadata
    metadata = {
        "architecture": architecture,
        "n_concepts": n_concepts,
        "num_classes": params["num_classes"],
        "concept_names": concept_names,
        "concept_weight": concept_weight,
        "freeze_backbone": freeze_backbone,
    }
    with open(os.path.join(save_dir, "cbm_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("Stage 1b (CBM) hoàn thành. Checkpoint tại %s", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
