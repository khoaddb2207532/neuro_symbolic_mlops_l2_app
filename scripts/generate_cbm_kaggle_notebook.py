"""Sinh notebook mỏng (launcher) chạy CBM + Rule + GFlowNet trên Kaggle.

Cách dùng:
  # Sinh 1 notebook mặc định (mobilenetv3_small, seed 42)
  python scripts/generate_cbm_kaggle_notebook.py

  # Sinh cho backbone và seed cụ thể
  python scripts/generate_cbm_kaggle_notebook.py --backbone resnet50 --seed 42 --output-dir generated_cbm_notebooks

  # Sinh hàng loạt cho tất cả các seeds (42, 44, 46, 48, 50)
  python scripts/generate_cbm_kaggle_notebook.py --all-seeds
"""
import argparse
import json
from pathlib import Path

DEFAULT_TEMPLATE = Path("cbm-neuro-symbolic-kaggle-template.ipynb")
DEFAULT_DATASET = "/kaggle/input/vietnamese-cultural-dataset/vietnamese_cultural_dataset"
SEEDS = [42, 44, 46, 48, 50]


def generate_notebook(
    template_path: Path,
    output_path: Path,
    backbone: str = "mobilenetv3_small",
    seed: int = 42,
    data_dir: str = DEFAULT_DATASET,
    n_concepts: int = 30,
    concept_weight: float = 1.0,
    num_epochs: int = 50,
    patience: int = 5,
    git_repo: str = "https://github.com/khoaddb2207532/neuro_symbolic_mlops_l2_app.git",
    git_branch: str = "main",
) -> Path:
    if not template_path.exists():
        raise FileNotFoundError(f"Không tìm thấy template tại: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Tìm cell parameters và cập nhật giá trị
    for cell in nb["cells"]:
        tags = cell.get("metadata", {}).get("tags", [])
        if "parameters" in tags:
            cell["source"] = [
                "# =========================================================================\n",
                "# THAM SỐ THỰC NGHIỆM TỰ ĐỘNG SINH BỞI GENERATOR SCRIPT\n",
                "# =========================================================================\n",
                f"DATA_DIR = {data_dir!r}\n",
                f"SEED = {seed}\n",
                f"BACKBONE = {backbone!r}\n",
                f"N_CONCEPTS = {n_concepts}\n",
                f"CONCEPT_WEIGHT = {concept_weight}\n",
                f"NUM_EPOCHS = {num_epochs}\n",
                f"PATIENCE = {patience}\n",
                "BATCH_SIZE = 64\n",
                f"GIT_REPO = {git_repo!r}\n",
                f"GIT_BRANCH = {git_branch!r}\n",
            ]
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Đã sinh notebook: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Sinh notebook mỏng CBM cho Kaggle.")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--output-dir", default="generated_cbm_notebooks")
    parser.add_argument("--backbone", default="mobilenetv3_small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=DEFAULT_DATASET)
    parser.add_argument("--n-concepts", type=int, default=30)
    parser.add_argument("--concept-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--git-repo", default="https://github.com/khoaddb2207532/neuro_symbolic_mlops_l2_app.git")
    parser.add_argument("--git-branch", default="main")
    parser.add_argument("--all-seeds", action="store_true", help="Sinh notebook cho tất cả các seeds (42, 44, 46, 48, 50)")

    args = parser.parse_args()
    template_path = Path(args.template)
    out_dir = Path(args.output_dir)

    seeds_to_run = SEEDS if args.all_seeds else [args.seed]

    for s in seeds_to_run:
        out_file = out_dir / f"cbm_neuro_symbolic_{args.backbone}_seed_{s}.ipynb"
        generate_notebook(
            template_path=template_path,
            output_path=out_file,
            backbone=args.backbone,
            seed=s,
            data_dir=args.data_dir,
            n_concepts=args.n_concepts,
            concept_weight=args.concept_weight,
            num_epochs=args.epochs,
            patience=args.patience,
            git_repo=args.git_repo,
            git_branch=args.git_branch,
        )


if __name__ == "__main__":
    main()
