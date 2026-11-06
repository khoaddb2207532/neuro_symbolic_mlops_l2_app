"""Stage 5C — train CNN với Bayesian marginalization từ EliteTracker states.

Stage này không rebuild hoặc sample GFlowNet policy. Nó ưu tiên state artifact
đã có trong ``output_dir`` (sau restore), sau đó kiểm tra trực tiếp các input
root, bao gồm ``/kaggle/input``. State được xác thực với rule order trước train.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from src.data.dataset import NeuroSymbolicDataset, create_dataloaders
from src.evaluation.evaluate import evaluate_model_performance, plot_training_history
from src.gflownet.rule_ranking_analysis import load_rule_order
from src.models.cnn import ImageClassificationBaseline
from src.rules.elite_bayesian_penalty import (
    EliteStateBayesianMarginalization,
    find_elite_states,
    load_elite_states,
)
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


def _search_roots(params: dict, filtered_dir: Path, config_path: Path) -> list[Path]:
    cfg = params.get("rule_penalty_bayesian_elite", {})
    configured = cfg.get("input_roots", [])
    if isinstance(configured, (str, os.PathLike)):
        configured = [configured]
    roots = [filtered_dir, *[Path(item) for item in configured]]
    kaggle_input = Path(os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input"))
    roots.extend([kaggle_input, config_path.parent])
    # Stable de-duplication avoids scanning the same restored tree repeatedly.
    return list(dict.fromkeys(path.expanduser() for path in roots))


def main(params_path: str) -> None:
    config_path = Path(params_path).expanduser().resolve()
    params = load_params(str(config_path))
    set_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(params["output_dir"]).expanduser()
    filtered_dir = output_dir / "04_filtered_rules"
    rule_order = load_rule_order(str(filtered_dir))
    valid_rules = list(rule_order["valid_rules"])

    elite_cfg = params.get("rule_penalty_bayesian_elite", {})
    states_path = find_elite_states(
        elite_cfg.get("states_path"),
        _search_roots(params, filtered_dir, config_path),
    )
    states, log_rewards, state_metadata = load_elite_states(
        states_path, valid_rules
    )
    logger.info(
        "Đã nạp %d elite states (%d luật/state) từ %s%s.",
        states.shape[0],
        states.shape[1],
        states_path,
        " với log-reward" if log_rewards is not None else " (uniform)",
    )

    penalty_cfg = params["rule_penalty"]
    penalty_module = EliteStateBayesianMarginalization(
        valid_rules=valid_rules,
        states=states,
        log_rewards=log_rewards,
        reward_temperature=float(elite_cfg.get("reward_temperature", 1.0)),
        K=int(elite_cfg.get("K", 32)),
        penalty_weight=penalty_cfg["weight"],
        use_confidence=True,
        smoothing=penalty_cfg["smoothing"],
        num_classes=params["num_classes"],
        initial_temp=penalty_cfg["initial_temp"],
        final_temp=penalty_cfg["final_temp"],
        temp_warmup_epochs=penalty_cfg.get("temp_warmup_epochs", 2),
        temp_anneal_epochs=penalty_cfg.get("temp_anneal_epochs", 10),
    )

    _, train_loader, val_loader, test_loader = create_dataloaders(
        params["data_dir"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        seed=params["seed"],
    )
    class_names = [
        name
        for name, _ in sorted(
            NeuroSymbolicDataset(params["data_dir"], "test").class_to_idx.items(),
            key=lambda item: item[1],
        )
    ]

    save_dir = output_dir / "05c_rules_model_bayesian_elite"
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "elite_state_source.json").write_text(
        json.dumps(
            {
                **state_metadata,
                "K": int(elite_cfg.get("K", 32)),
                "reward_temperature": float(
                    elite_cfg.get("reward_temperature", 1.0)
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    architecture = selected_baseline_architecture(params)
    model = ImageClassificationBaseline(
        architecture=architecture,
        num_classes=params["num_classes"],
        pretrained=False,
    )
    load_model_weights(
        model, selected_baseline_checkpoint(params), device, required=True
    )

    train_cfg = {
        "lr_backbone": params["transfer_learning"]["lr_backbone"],
        "lr_head": params["transfer_learning"]["lr_head"],
        "weight_decay": params["weight_decay"],
        "monitor_metric": params.get("monitor_metric", "val_acc"),
        "dvclive_path": str(save_dir / "dvclive_rule_regularized_bayesian_elite"),
        "save_dir": str(save_dir),
    }
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        rule_set=None,
        penalty_module=penalty_module,
        num_epochs=params["num_epochs"],
        patience=params["patience"],
        device=device,
        penalty_weight=penalty_cfg["weight"],
        use_confidence=True,
        smoothing=penalty_cfg["smoothing"],
        initial_temp=penalty_cfg["initial_temp"],
        final_temp=penalty_cfg["final_temp"],
        temp_warmup_epochs=penalty_cfg.get("temp_warmup_epochs", 2),
        temp_anneal_epochs=penalty_cfg.get("temp_anneal_epochs", 10),
        min_epochs_before_early_stop=penalty_cfg.get(
            "min_epochs_before_early_stop", 12
        ),
        num_classes=params["num_classes"],
        config=train_cfg,
    )

    sampling_summary = penalty_module.save_sampling_summary(str(save_dir))
    evaluate_model_performance(
        model,
        test_loader,
        device,
        class_names,
        title="Rule-Regularized CNN (Bayesian Elite States)",
        output_dir=str(save_dir),
    )
    plot_training_history(
        history,
        save_dir=str(save_dir),
        title_suffix="Rule-Regularized CNN (Bayesian Elite States)",
    )
    logger.info(
        "Stage 5C hoàn thành: %d ruleset samples từ %d elite states.",
        sampling_summary["total_sampled_rulesets"],
        states.shape[0],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    main(parser.parse_args().config)

