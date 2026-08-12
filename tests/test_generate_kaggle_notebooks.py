import json
from pathlib import Path

from scripts.generate_kaggle_notebooks import generate, generate_matrix


def test_dataset_b_gets_two_cells_after_clone(tmp_path: Path) -> None:
    registry = tmp_path / "registry.csv"
    registry.write_text(
        "run_id,dataset_id,data_dir,seed,backbone,account_id\n"
        "run-a,culture-a,/data/a,42,resnet50,account-1\n"
        "run-b,culture-b,/kaggle/working,42,resnet50,account-3\n",
        encoding="utf-8",
    )
    template = Path("managed-experiment-runner-template.ipynb")
    generate(registry, template, tmp_path / "generated", "abc123")
    notebook_a = json.loads(
        (tmp_path / "generated/resnet50/run-a.ipynb").read_text(encoding="utf-8")
    )
    notebook_b = json.loads(
        (tmp_path / "generated/resnet50/run-b.ipynb").read_text(encoding="utf-8")
    )

    tags_a = [tag for cell in notebook_a["cells"] for tag in cell.get("metadata", {}).get("tags", [])]
    tags_b = [tag for cell in notebook_b["cells"] for tag in cell.get("metadata", {}).get("tags", [])]
    assert "dataset-b-warning-control" not in tags_a
    assert "dataset-b-symlinks" not in tags_a
    assert "dataset-b-warning-control" in tags_b
    assert "dataset-b-symlinks" in tags_b

    clone_index = next(
        index for index, cell in enumerate(notebook_b["cells"])
        if any("GITHUB_TOKEN" in line for line in cell.get("source", []))
    )
    assert "dataset-b-warning-control" in notebook_b["cells"][clone_index + 1]["metadata"]["tags"]
    assert "dataset-b-symlinks" in notebook_b["cells"][clone_index + 2]["metadata"]["tags"]


def test_generate_matrix_creates_twenty_core_prior_notebooks_without_csv(tmp_path: Path) -> None:
    generated = generate_matrix(
        Path("managed-experiment-runner-template.ipynb"),
        tmp_path / "generated",
        "abc123",
        backbones=["efficientnet_b0", "shufflenet_v2_x1_0"],
        seeds=[42, 44, 46, 48, 50],
        datasets=["culture-a", "culture-b"],
    )

    assert len(generated) == 20
    expected = (
        tmp_path
        / "generated"
        / "shufflenet_v2_x1_0"
        / "culture-b__shufflenet_v2_x1_0__db__seed_50.ipynb"
    )
    notebook = json.loads(expected.read_text(encoding="utf-8"))
    parameter_cell = next(
        cell
        for cell in notebook["cells"]
        if "experiment-parameters" in cell.get("metadata", {}).get("tags", [])
    )
    source = "".join(parameter_cell["source"])
    assert "BACKBONE = 'shufflenet_v2_x1_0'" in source
    assert "DATA_DIR = '/kaggle/working'" in source
    assert "RUN_ID = 'culture-b__shufflenet_v2_x1_0__db__seed_50'" in source
