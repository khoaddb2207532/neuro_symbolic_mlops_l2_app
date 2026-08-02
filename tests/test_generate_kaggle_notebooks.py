import json
from pathlib import Path

from scripts.generate_kaggle_notebooks import generate


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
    notebook_a = json.loads((tmp_path / "generated/run-a.ipynb").read_text(encoding="utf-8"))
    notebook_b = json.loads((tmp_path / "generated/run-b.ipynb").read_text(encoding="utf-8"))

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
