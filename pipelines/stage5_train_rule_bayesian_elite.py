"""Stage 5 Bayesian using the non-convergence-dependent elite GFlowNet.

The implementation is shared with ``stage5_train_rule_bayesian``; this entry
point fixes the variant to ``elite`` so DVC/Kaggle commands cannot silently use
the diverse or converged sampler.
"""

import argparse

from pipelines.stage5_train_rule_bayesian import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config, variant="elite")
