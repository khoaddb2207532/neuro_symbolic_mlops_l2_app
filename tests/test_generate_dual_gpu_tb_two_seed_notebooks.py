import ast
import json
from pathlib import Path

import pytest

from pipelines.run_dual_gpu_tb_multi_seed_experiment import _resume_enabled, partition_runs
from scripts.generate_dual_gpu_tb_two_seed_notebooks import (
    generate_two_seed_notebooks,
    validate_seeds,
)


def _runs_from_notebook(path: Path) -> list[dict]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cell = next(
        item
        for item in notebook["cells"]
        if "tb-multiseed-parameters" in item.get("metadata", {}).get("tags", [])
    )
    line = next(line for line in cell["source"] if line.startswith("RUNS = "))
    return ast.literal_eval(line.removeprefix("RUNS = ").strip())


def test_validate_seeds_requires_two_distinct_values():
    assert validate_seeds((46, 50)) == (46, 50)
    with pytest.raises(ValueError, match="đúng 2 seed"):
        validate_seeds((46,))
    with pytest.raises(ValueError, match="khác nhau"):
        validate_seeds((46, 46))


def test_runner_defaults_to_resume_but_honors_fresh_run_marker():
    assert _resume_enabled({"seed": 42})
    assert _resume_enabled({"seed": 42, "resume": True})
    assert not _resume_enabled({"seed": 46, "resume": False})


def test_generated_notebook_assigns_one_seed_per_gpu_across_datasets(tmp_path: Path):
    registry = tmp_path / "registry.csv"
    registry.write_text(
        "run_id,dataset_id,data_dir,seed,backbone\n"
        "a,culture-a,/data/a,42,vit_b_32\n"
        "b,culture-b,/data/b,42,vit_b_32\n",
        encoding="utf-8",
    )
    [path] = generate_two_seed_notebooks(
        registry,
        Path("dual-gpu-tb-multiseed-template.ipynb"),
        tmp_path / "notebooks",
        git_ref="main",
        seeds=(46, 50),
        backbones=("vit_b_32",),
    )

    assert path.name == "dual_gpu_tb_two_seed_46_50_vit_b_32.ipynb"
    queues = partition_runs(_runs_from_notebook(path))
    assert [{run["seed"] for run in queue} for queue in queues] == [{46}, {50}]
    assert all(not run["resume"] for queue in queues for run in queue)
    source = path.read_text(encoding="utf-8")
    assert "GPU 0 chỉ chạy seed `46`" in source
    assert "GPU 1 chỉ chạy seed `50`" in source
    assert "Resume output cũ: `tắt`" in source
