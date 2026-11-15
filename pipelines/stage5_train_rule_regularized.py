"""Stage 5 — Fine-tune lại mô hình với rule penalty (vectorized) từ luật đã chọn.

Hai đích (``rule_penalty.target_model``):
  - ``cbm``: dùng lại ``train_cbm`` (src/training/cbm_trainer.py) nên loss giữ đủ ba phần
        Loss = CE + λ_concept·BCE(concept) + λ_rule·RulePenalty
    LR nhỏ (``rule_penalty.lr_backbone`` / ``lr_head``, mặc định 1e-5 / 1e-4), ``penalty_off_epochs``
    epoch đầu tắt penalty rồi bật với T khởi đầu ``initial_temp`` (xem params.yaml).
  - ``baseline``: như cũ, ``train_model`` với CE + penalty, khởi động từ baseline_best.pth
    (KHÔNG khởi tạo lại từ ImageNet).
"""
import argparse
import hashlib
import json
import os
import pickle
from typing import Any, cast

import torch

from pipelines.stage1b_train_cbm import resolve_pos_weight, wrap_with_concepts
from src.data.dataset import create_dataloaders, NeuroSymbolicDataset
from src.evaluation.evaluate import evaluate_model_performance, plot_training_history
from src.models.cbm import ConceptBottleneckModel
from src.models.cnn import ImageClassificationBaseline
from src.rules.penalty import VectorizedRulePenalty
from src.rules.rule_types import RuleSet
from src.training.cbm_trainer import train_cbm
from src.training.trainer import train_model
from src.utils.checkpoint import load_model_weights
from src.utils.config import (
    load_params,
    selected_baseline_architecture,
    selected_baseline_checkpoint,
)
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: str) -> dict[str, Any]:
    """Đọc JSON và bảo đảm gốc là object (ranh giới kiểu duy nhất của file này)."""
    with open(path, "r", encoding="utf-8") as file:
        raw: object = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} phải chứa một JSON object, nhận {type(raw).__name__}.")
    return cast(dict[str, Any], raw)


class _CBMEvalAdapter(torch.nn.Module):
    """evaluate_model_performance cần model(x) -> (logits, features); CBM trả thêm raw_features."""

    def __init__(self, cbm: torch.nn.Module) -> None:
        super().__init__()
        self.cbm = cbm

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, concept_probs, _ = self.cbm(x)
        return logits, concept_probs


def _train_cbm_with_rules(
    params: dict[str, Any],
    rule_set: RuleSet,
    selected_rules_sha256: str,
    dataloaders: dict[str, Any],
    test_loader: Any,
    class_names: list[str],
    save_dir: str,
    device: str,
    rule_weight_override: float | None = None,
) -> None:
    """Fine-tune CBM bằng ``train_cbm``: CE + λ_concept·BCE + rule penalty, LR nhỏ.

    ``rule_weight_override`` thay ``rule_penalty.weight`` (đối chứng CBM-continued dùng 0.0).
    Mọi tham số khác (LR, epoch, penalty_off_epochs, lịch nhiệt độ) giữ nguyên để đối chứng
    chỉ khác đúng một điểm: có hay không luật.
    """
    rule_cfg: dict[str, Any] = dict(params["rule_penalty"])
    if rule_weight_override is not None:
        if rule_weight_override < 0:
            raise ValueError("rule_weight_override không được âm.")
        rule_cfg["weight"] = float(rule_weight_override)
    cbm_cfg: dict[str, Any] = params.get("cbm", {})
    architecture = selected_baseline_architecture(params)

    cbm_dir = os.path.join(params["output_dir"], "cbm")
    metadata_path = os.path.join(cbm_dir, "cbm_metadata.json")
    meta: dict[str, Any] = _load_json_object(metadata_path) if os.path.exists(metadata_path) else {}
    arch = str(meta.get("architecture", architecture))
    n_concepts = int(meta.get("n_concepts", cbm_cfg.get("n_concepts", 30)))

    # Dataloader có concept target (cùng nguồn nhãn với Stage 1b) để giữ được concept loss.
    wrapped, concept_names, wrapped_n_concepts = wrap_with_concepts(
        dataloaders, params, class_names, cbm_dir
    )
    if wrapped_n_concepts != n_concepts:
        raise ValueError(
            f"Số concept trong annotations ({wrapped_n_concepts}) khác checkpoint CBM "
            f"({n_concepts}); luật Stage 3 đánh chỉ số theo concept nên không được lệch."
        )
    concept_pos_weight = resolve_pos_weight(cbm_cfg, wrapped["train"].dataset)

    cbm_ckpt = os.path.join(cbm_dir, "cbm_best.pth")
    if not os.path.exists(cbm_ckpt):
        cbm_ckpt = os.path.join(cbm_dir, "final_model_weights.pth")
    model = ConceptBottleneckModel(
        architecture=arch,
        n_concepts=n_concepts,
        num_classes=params["num_classes"],
        pretrained=False,
        concept_names=concept_names,
    )
    load_model_weights(model, cbm_ckpt, device, required=True)

    lr_backbone = float(rule_cfg.get("lr_backbone", 1e-5))
    lr_head = float(rule_cfg.get("lr_head", 1e-4))
    concept_weight = float(rule_cfg.get("concept_weight", cbm_cfg.get("concept_weight", 1.0)))
    off_epochs = int(rule_cfg.get("penalty_off_epochs", 0))
    # Early stopping chỉ được tính sau khi penalty đã bật đủ min_epochs epoch.
    min_epochs = off_epochs + int(rule_cfg.get("min_epochs_before_early_stop", 12))

    penalty_module = VectorizedRulePenalty(
        rule_set,
        penalty_weight=float(rule_cfg["weight"]),
        use_confidence=True,
        smoothing=float(rule_cfg["smoothing"]),
        num_classes=int(params["num_classes"]),
        initial_temp=float(rule_cfg["initial_temp"]),
        final_temp=float(rule_cfg["final_temp"]),
        temp_warmup_epochs=int(rule_cfg.get("temp_warmup_epochs", 2)),
        temp_anneal_epochs=int(rule_cfg.get("temp_anneal_epochs", 10)),
    )
    logger.info(
        "Stage 5 (CBM): %d luật | λ_concept=%.2f λ_rule=%.3f | lr_backbone=%.1e lr_head=%.1e | "
        "tắt penalty %d epoch đầu, T %.1f→%.1f | từ %s",
        len(rule_set), concept_weight, float(rule_cfg["weight"]), lr_backbone, lr_head,
        off_epochs, float(rule_cfg["initial_temp"]), float(rule_cfg["final_temp"]), cbm_ckpt,
    )

    train_cfg: dict[str, Any] = {
        "lr_backbone": lr_backbone,
        "lr_head": lr_head,
        "weight_decay": params["weight_decay"],
        "monitor_metric": params.get("monitor_metric", "val_loss"),
        "save_dir": save_dir,
    }
    model, history = train_cbm(
        model=model,
        train_loader=wrapped["train"],
        val_loader=wrapped["val"],
        num_epochs=params["num_epochs"],
        lr=lr_head,
        patience=params["patience"],
        device=device,
        concept_weight=concept_weight,
        smoothing=float(rule_cfg["smoothing"]),
        num_classes=int(params["num_classes"]),
        min_epochs_before_early_stop=min_epochs,
        config=train_cfg,
        concept_pos_weight=concept_pos_weight,
        two_stage=False,
        penalty_module=penalty_module,
        penalty_off_epochs=off_epochs,
        ckpt_filename="rule_regularized_best.pth",
    )

    with open(os.path.join(save_dir, "history.json"), "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "rule_regularized_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "architecture": arch,
                "n_concepts": n_concepts,
                "n_rules": len(rule_set),
                "selected_rules_sha256": selected_rules_sha256,
                "concept_weight": concept_weight,
                "lr_backbone": lr_backbone,
                "lr_head": lr_head,
                "penalty_off_epochs": off_epochs,
                "rule_weight_override": rule_weight_override,
                "rule_penalty": dict(rule_cfg),
            },
            file, ensure_ascii=False, indent=2,
        )

    evaluate_model_performance(
        _CBMEvalAdapter(model), test_loader, device, class_names,
        title="Rule-Regularized CBM Performance", output_dir=save_dir,
    )
    plot_training_history(history, save_dir=save_dir, title_suffix="Rule-Regularized CBM")


def main(
    params_path: str,
    target_override: str | None = None,
    rule_weight_override: float | None = None,
    save_subdir: str = "05_rules_model",
) -> None:
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

    save_dir = os.path.join(params["output_dir"], save_subdir)
    os.makedirs(save_dir, exist_ok=True)

    target_model = target_override or params.get("rule_penalty", {}).get("target_model", "baseline")
    architecture = selected_baseline_architecture(params)

    if target_model == "cbm":
        _train_cbm_with_rules(
            params, rule_set, _sha256_file(os.path.join(filtered_dir, "selected_rules.pkl")),
            dataloaders, test_loader, class_names, save_dir, device,
            rule_weight_override=rule_weight_override,
        )
        logger.info("Stage 5 hoàn thành (CBM + rule penalty).")
        return

    model = ImageClassificationBaseline(
        architecture=architecture,
        num_classes=params["num_classes"],
        pretrained=False,
    )
    baseline_ckpt = selected_baseline_checkpoint(params)
    load_model_weights(model, baseline_ckpt, device, required=True)
    logger.info("Fine-tune Baseline CNN với Rule Penalty từ %s", baseline_ckpt)

    train_cfg = {
        "lr_backbone": params["transfer_learning"]["lr_backbone"],
        "lr_head": params["transfer_learning"]["lr_head"],
        "weight_decay": params["weight_decay"],
        "monitor_metric": params.get("monitor_metric", "val_acc"),
        "save_dir": save_dir,
        "resume": bool(params["rule_penalty"].get("resume", False)),
        "resume_checkpoint_path": os.path.join(save_dir, "training_last.pth"),
        "resume_identity": {
            "stage": "stage5_train_rule_regularized",
            "seed": params["seed"],
            "architecture": architecture,
            "gflownet_loss_type": params.get("gflownet", {}).get("loss_type"),
            "selected_rules_sha256": _sha256_file(
                os.path.join(filtered_dir, "selected_rules.pkl")
            ),
            "batch_size": params["batch_size"],
            "optimizer": {
                "lr_backbone": params["transfer_learning"]["lr_backbone"],
                "lr_head": params["transfer_learning"]["lr_head"],
                "weight_decay": params["weight_decay"],
            },
            "rule_penalty": dict(params["rule_penalty"]),
        },
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
        temp_warmup_epochs=params["rule_penalty"].get("temp_warmup_epochs", 2),
        temp_anneal_epochs=params["rule_penalty"].get("temp_anneal_epochs", 10),
        min_epochs_before_early_stop=params["rule_penalty"].get("min_epochs_before_early_stop", 12),
        num_classes=params["num_classes"],
        config=train_cfg,
    )

    evaluate_model_performance(model, test_loader, device, class_names, title="Rule-Regularized CNN Performance", output_dir=save_dir)
    plot_training_history(history, save_dir=save_dir, title_suffix="Rule-Regularized CNN")
    logger.info("Stage 5 hoàn thành với baseline '%s'.", architecture)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument(
        "--target",
        choices=["baseline", "cbm"],
        default=None,
        help="Model cần fine-tune với luật: 'baseline' hoặc 'cbm'",
    )
    parser.add_argument(
        "--rule-weight",
        type=float,
        default=None,
        help="Ghi đè rule_penalty.weight (đối chứng CBM-continued dùng 0)",
    )
    parser.add_argument(
        "--save-subdir",
        default="05_rules_model",
        help="Thư mục con trong output_dir để lưu kết quả (tránh ghi đè bản chính)",
    )
    args = parser.parse_args()
    main(
        args.config,
        target_override=args.target,
        rule_weight_override=args.rule_weight,
        save_subdir=args.save_subdir,
    )
