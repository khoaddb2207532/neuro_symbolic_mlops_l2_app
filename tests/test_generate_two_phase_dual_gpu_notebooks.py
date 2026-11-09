import json
from pathlib import Path

from scripts.generate_two_phase_dual_gpu_notebooks import generate


def test_generates_seed_pair_priors_and_per_seed_db_stage5(tmp_path: Path):
    priors, stage5 = generate(
        tmp_path,
        "a" * 40,
        backbones=["efficientnet_b0", "shufflenet_v2_x1_0"],
        datasets=["culture-a", "culture-b"],
        seeds=[42, 44, 46, 48, 50],
        seeds_per_prior_notebook=2,
    )
    assert len(priors) == 10
    assert len(stage5) == 20
    prior = json.loads(priors[0].read_text(encoding="utf-8"))
    prior_source = "".join(cell_source for cell in prior["cells"] for cell_source in cell["source"])
    assert "pipelines.run_dual_gpu_stage1_4_multi_seed" in prior_source
    assert "'seed': 42" in prior_source and "'seed': 44" in prior_source
    assert "DISABLE_DVCLIVE" in prior_source
    assert "DVCLive disabled" in prior_source
    final = json.loads(stage5[-1].read_text(encoding="utf-8"))
    final_source = "".join(cell_source for cell in final["cells"] for cell_source in cell["source"])
    assert "pipelines.run_dual_gpu_db_stage5_seed" in final_source
    assert "shufflenet_v2_x1_0" in final_source
    assert "TB" not in final_source
    assert "DISABLE_DVCLIVE" in final_source
