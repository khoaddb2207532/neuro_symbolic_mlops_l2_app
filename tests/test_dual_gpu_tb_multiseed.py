import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines.run_dual_gpu_tb_multi_seed_experiment import _aggregate, partition_runs
from scripts.generate_dual_gpu_tb_multiseed_notebooks import generate


def test_partition_runs_uses_two_stable_round_robin_queues():
    runs = [{"seed": seed} for seed in (42, 44, 46, 48, 50)]
    queues = partition_runs(runs)
    assert [[run["seed"] for run in queue] for queue in queues] == [
        [42, 46, 50],
        [44, 48],
    ]


def test_generator_includes_shufflenet_vit_b32_swin_and_tb_stages(tmp_path: Path):
    registry = tmp_path / "registry.csv"
    registry.write_text(
        "run_id,dataset_id,data_dir,seed,backbone\n"
        "a,culture-a,/data,42,alexnet\n",
        encoding="utf-8",
    )
    generated = generate(
        registry,
        Path("dual-gpu-tb-multiseed-template.ipynb"),
        tmp_path / "notebooks",
        git_ref="main",
        backbones=("shufflenet_v2_x1_0", "vit_b_32", "swin_t"),
        seeds=(42, 44),
        datasets=("culture-a",),
    )
    assert [path.name for path in generated] == [
        "dual_gpu_tb_multiseed_shufflenet_v2_x1_0.ipynb",
        "dual_gpu_tb_multiseed_vit_b_32.ipynb",
        "dual_gpu_tb_multiseed_swin_t.ipynb",
    ]
    source = json.dumps(json.loads(generated[0].read_text(encoding="utf-8")))
    assert "pipelines.run_dual_gpu_tb_multi_seed_experiment" in source
    assert "comparison-table" in source
    assert "shufflenet_v2_x1_0" in source


def test_aggregate_writes_detail_and_mean_std_tables(tmp_path: Path):
    paths = []
    for seed, accuracy in ((42, 0.8), (44, 0.9)):
        path = tmp_path / f"seed_{seed}.csv"
        pd.DataFrame(
            [
                {
                    "dataset_id": "culture-a",
                    "backbone": "swin_t",
                    "seed": seed,
                    "method": "gflownet_tb_fixed",
                    "test_accuracy": accuracy,
                    "test_f1_macro": accuracy - 0.1,
                    "test_macro_precision": accuracy,
                    "test_macro_recall": accuracy,
                    "accuracy_delta_vs_baseline": 0.01,
                    "f1_delta_vs_baseline": 0.02,
                }
            ]
        ).to_csv(path, index=False)
        paths.append(path)
    detail, summary = _aggregate(tmp_path, paths)
    assert detail.exists()
    row = pd.read_csv(summary).iloc[0]
    assert row["n_seeds"] == 2
    assert row["test_accuracy_mean"] == pytest.approx(0.85)
    assert row["test_accuracy_std"] == pytest.approx(2**-0.5 * 0.1)
    assert (tmp_path / "tb_all_seed_summary.md").exists()
