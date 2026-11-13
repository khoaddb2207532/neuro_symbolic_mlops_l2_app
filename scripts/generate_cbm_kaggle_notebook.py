"""Sinh notebook mỏng (launcher) Dual-GPU T4 x2 chạy CBM + Rule + GFlowNet trên Kaggle.

Cách dùng:
  # Sinh 1 notebook Dual-GPU mặc định (Seed 42 trên GPU 0, Seed 44 trên GPU 1)
  python scripts/generate_cbm_kaggle_notebook.py

  # Sinh cho cặp seeds khác hoặc backbone khác
  python scripts/generate_cbm_kaggle_notebook.py --backbone resnet50 --seeds 46 48
"""
import argparse
import json
from pathlib import Path

DEFAULT_TEMPLATE = Path("cbm-neuro-symbolic-kaggle-template.ipynb")
DEFAULT_DATASET = "/kaggle/input/datasets/dangduykhoab2207532/mykhmer/khmer"
DEFAULT_CONCEPT_PATH = "/kaggle/input/datasets/dangduykhoab2207532/mykhmer/image_concepts.json"


def generate_dual_gpu_notebook(
    template_path: Path,
    output_path: Path,
    backbone: str = "mobilenetv3_small",
    seed_a: int = 42,
    seed_b: int = 44,
    data_dir: str = DEFAULT_DATASET,
    concept_annotations_path: str = DEFAULT_CONCEPT_PATH,
    num_classes: int = 9,
    n_concepts: int = 61,
    concept_weight: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 2,
    num_epochs: int = 50,
    patience: int = 5,
    git_repo: str = "https://github.com/khoaddb2207532/neuro_symbolic_mlops_l2_app.git",
    git_branch: str = "cbm",
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
                "# THAM SỐ THỰC NGHIỆM DUAL-GPU (Tự động sinh bởi generator script)\n",
                "# =========================================================================\n",
                f"DATA_DIR = {data_dir!r}\n",
                f"CONCEPT_ANNOTATIONS_PATH = {concept_annotations_path!r}\n",
                f"NUM_CLASSES = {num_classes}\n",
                f"N_CONCEPTS = {n_concepts}\n",
                f"BACKBONE = {backbone!r}\n",
                "\n",
                f"SEED_A = {seed_a}                     # Chạy trên GPU 0 (15GB VRAM)\n",
                f"SEED_B = {seed_b}                     # Chạy trên GPU 1 (15GB VRAM)\n",
                "\n",
                f"BATCH_SIZE = {batch_size}                 # Chiếm ~2.5GB VRAM (an toàn cho 15GB VRAM)\n",
                f"NUM_WORKERS = {num_workers}                 # 2 workers mỗi GPU (tổng 4 workers < 6GB RAM / 30GB)\n",
                f"NUM_EPOCHS = {num_epochs}\n",
                f"PATIENCE = {patience}\n",
                f"CONCEPT_WEIGHT = {concept_weight}\n",
                f"GIT_REPO = {git_repo!r}\n",
                f"GIT_BRANCH = {git_branch!r}\n",
            ]
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Đã sinh notebook Dual-GPU: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Sinh notebook mỏng CBM Dual-GPU cho Kaggle.")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--output-dir", default="generated_cbm_notebooks")
    parser.add_argument("--backbone", default="mobilenetv3_small")
    parser.add_argument("--seeds", nargs=2, type=int, default=[42, 44], help="2 seeds chạy song song (vd: 42 44)")
    parser.add_argument("--data-dir", default=DEFAULT_DATASET)
    parser.add_argument("--concept-path", default=DEFAULT_CONCEPT_PATH)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--n-concepts", type=int, default=61)
    parser.add_argument("--concept-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--git-repo", default="https://github.com/khoaddb2207532/neuro_symbolic_mlops_l2_app.git")
    parser.add_argument("--git-branch", default="cbm")

    args = parser.parse_args()
    template_path = Path(args.template)
    out_dir = Path(args.output_dir)

    out_file = out_dir / f"dual_gpu_cbm_{args.backbone}_seeds_{args.seeds[0]}_{args.seeds[1]}.ipynb"
    generate_dual_gpu_notebook(
        template_path=template_path,
        output_path=out_file,
        backbone=args.backbone,
        seed_a=args.seeds[0],
        seed_b=args.seeds[1],
        data_dir=args.data_dir,
        concept_annotations_path=args.concept_path,
        num_classes=args.num_classes,
        n_concepts=args.n_concepts,
        concept_weight=args.concept_weight,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.epochs,
        patience=args.patience,
        git_repo=args.git_repo,
        git_branch=args.git_branch,
    )


if __name__ == "__main__":
    main()
