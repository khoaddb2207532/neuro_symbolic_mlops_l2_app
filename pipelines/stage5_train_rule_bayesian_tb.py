"""Stage 5 Bayesian using a frozen GFlowNet trained with TB objective.

This strict entry point refuses DB/FM rule-order artifacts, preventing a
notebook/config mix-up from being mislabeled as the TB Bayesian experiment.
"""

import argparse
import os

from src.gflownet.rule_ranking_analysis import load_rule_order
from src.utils.config import load_params

from pipelines.stage5_train_rule_bayesian import main


def run(params_path: str) -> None:
    params = load_params(params_path)
    filtered_dir = os.path.join(params["output_dir"], "04_filtered_rules")
    rule_order = load_rule_order(filtered_dir)
    loss_type = rule_order.get("loss_type")
    if loss_type != "tb":
        raise ValueError(
            "stage5_train_rule_bayesian_tb yêu cầu GFlowNet loss_type='tb', "
            f"nhưng rule order chứa loss_type={loss_type!r}."
        )
    main(params_path, variant="diverse")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    run(args.config)
