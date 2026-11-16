"""Sinh notebook Kaggle (T4 x2) CHẠY TIẾP từ kết quả lần chạy trước.

Notebook không train lại Baseline/CBM/RF/GFlowNet cũ. Nó:
  1. Tìm ``outputs_seed_<seed>/`` của lần chạy trước trong ``/kaggle/input`` (hoặc zip
     ``*outputs*.zip``), kiểm tra backbone và số concept khớp notebook.
  2. Khôi phục vào repo, rồi chạy Stage 4C (GFlowNet xây luật từ image_concept) và
     Stage 5 ở ba mode penalty / head / both trên 2 GPU song song.
  3. Xuất ``concept_rules_outputs.zip`` gồm luật mới và bảng so sánh.

Cách dùng:
  python scripts/generate_concept_rules_kaggle_notebook.py                       # 1 notebook mặc định
  python scripts/generate_concept_rules_kaggle_notebook.py --backbone resnet50 --seeds 46 48
  python scripts/generate_concept_rules_kaggle_notebook.py --all                 # mọi backbone x cặp seed

Notebook được ghi vào ``generated_concept_rules_notebooks/`` (tách khỏi notebook CBM gốc).
Đẩy lên Kaggle:
  python scripts/push_kaggle.py --notebook-dir generated_concept_rules_notebooks \\
      --extra-dataset <owner>/<dataset-chứa-kết-quả-lần-chạy-trước>
"""
import argparse
import json
from pathlib import Path

TEMPLATE = Path("cbm-neuro-symbolic-kaggle-template.ipynb")
DEFAULT_OUTPUT_DIR = Path("generated_concept_rules_notebooks")
DEFAULT_DATASET = "/kaggle/input/datasets/dangduykhoab2207532/mykhmer/khmer"
DEFAULT_CONCEPT_PATH = "/kaggle/input/datasets/dangduykhoab2207532/mykhmer/image_concepts.json"
DEFAULT_GIT_REPO = "https://github.com/khoaddb2207532/neuro_symbolic_mlops_l2_app.git"
DEFAULT_GIT_BRANCH = "cbm"

BACKBONES = ["alexnet", "mobilenetv3_small", "resnet50", "swin_t", "vit_b_32"]
SEED_PAIRS = [(42, 44), (46, 48)]

PARAMETERS_CELL = '''# =========================================================================
# THAM SỐ CHẠY TIẾP (sinh tự động bởi generate_concept_rules_kaggle_notebook.py)
# =========================================================================
DATA_DIR = {data_dir!r}
CONCEPT_ANNOTATIONS_PATH = {concept_path!r}
NUM_CLASSES = {num_classes}
N_CONCEPTS = {n_concepts}
BACKBONE = {backbone!r}
SEED_A = {seed_a}
SEED_B = {seed_b}
CONCEPT_WEIGHT = {concept_weight}
BATCH_SIZE = {batch_size}
NUM_WORKERS = {num_workers}
NUM_EPOCHS = {num_epochs}
PATIENCE = {patience}
MODES = {modes!r}          # penalty | head | both
GIT_REPO = {git_repo!r}
GIT_BRANCH = {git_branch!r}
'''

CHECK_DATA_CELL = '''# Kiểm tra dữ liệu ảnh và file nhãn concept (tự tìm lại nếu đường dẫn khác)
from pathlib import Path

data_path = Path(DATA_DIR)
concept_path = Path(CONCEPT_ANNOTATIONS_PATH)
if not data_path.exists():
    candidates = list(Path("/kaggle/input").glob("**/khmer"))
    if candidates:
        DATA_DIR = str(candidates[0])
        print(f"Tự động phát hiện DATA_DIR: {DATA_DIR}")
    else:
        raise FileNotFoundError(f"Không tìm thấy DATA_DIR: {DATA_DIR}")
if not concept_path.exists():
    candidates = list(Path("/kaggle/input").glob("**/*concept*.json"))
    if candidates:
        CONCEPT_ANNOTATIONS_PATH = str(candidates[0])
        print(f"Tự động phát hiện file concept: {CONCEPT_ANNOTATIONS_PATH}")
    else:
        raise FileNotFoundError(f"Không tìm thấy file concept: {CONCEPT_ANNOTATIONS_PATH}")
print("DATA_DIR:", DATA_DIR)
print("CONCEPT_ANNOTATIONS_PATH:", CONCEPT_ANNOTATIONS_PATH)
'''

CLONE_CELL = '''# Clone repo đúng branch và kiểm tra branch ĐÃ CÓ mã luật concept (patch đã được push)
import subprocess
from pathlib import Path

PROJECT = Path("/kaggle/working/neuro_symbolic_mlops_l2_app")
if not PROJECT.exists():
    subprocess.run(["git", "clone", "--depth", "1", "-b", GIT_BRANCH, GIT_REPO, str(PROJECT)], check=True)

required = [
    "src/gflownet/concept_rule_pipeline.py",
    "src/models/rule_head.py",
    "pipelines/stage4_concept_rules_gflownet.py",
    "pipelines/run_concept_rules_pipeline.py",
]
missing = [p for p in required if not (PROJECT / p).exists()]
if missing:
    raise RuntimeError(
        f"Branch '{GIT_BRANCH}' chưa có mã luật concept (thiếu {missing}). "
        "Hãy commit & push patch lên branch trước khi chạy notebook."
    )
print("Repo sẵn sàng tại:", PROJECT, "| đã có mã luật concept")
'''

INSTALL_CELL = '''# Cài đặt thư viện (giữ nguyên numpy của môi trường Kaggle)
%cd /kaggle/working/neuro_symbolic_mlops_l2_app
import importlib, importlib.metadata as md, subprocess, sys

print("numpy gốc:", md.version("numpy"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torchgfn", "tensordict", "openpyxl", "psutil"], check=True)
importlib.invalidate_caches()
for name in tuple(sys.modules):
    if name == "gfn" or name.startswith("gfn."):
        del sys.modules[name]

import torch
n_gpus = torch.cuda.device_count()
print("Số GPU:", n_gpus)
assert n_gpus >= 2, "HÃY CHỌN ACCELERATOR: GPU T4 x2 trên Kaggle!"
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    print(f"  [GPU {i}] {p.name} | VRAM {p.total_memory / 1024**3:.1f} GB")
'''

RESTORE_CELL = '''# Khôi phục kết quả lần chạy trước từ /kaggle/input (thư mục hoặc file zip).
import json, shutil, zipfile
from pathlib import Path

REPO = Path("/kaggle/working/neuro_symbolic_mlops_l2_app")
INPUT = Path("/kaggle/input")
EXTRACT_ROOT = Path("/kaggle/working/prev_extract")


def _matches(seed_dir: Path) -> bool:
    meta_path = seed_dir / "cbm" / "cbm_metadata.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("architecture") == BACKBONE and int(meta.get("n_concepts", -1)) == N_CONCEPTS


def find_previous(seed: int) -> Path | None:
    for cand in INPUT.glob(f"**/outputs_seed_{seed}"):
        if cand.is_dir() and _matches(cand):
            return cand
    # Không thấy thư mục: thử giải nén các zip kết quả rồi tìm lại
    if not EXTRACT_ROOT.exists():
        for z in INPUT.glob("**/*outputs*.zip"):
            target = EXTRACT_ROOT / z.stem
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(target)
    for cand in EXTRACT_ROOT.glob(f"**/outputs_seed_{seed}"):
        if cand.is_dir() and _matches(cand):
            return cand
    return None


restored = {}
for seed in (SEED_A, SEED_B):
    src = find_previous(seed)
    if src is None:
        raise FileNotFoundError(
            f"Không tìm thấy outputs_seed_{seed} của backbone '{BACKBONE}' ({N_CONCEPTS} concept) "
            "trong /kaggle/input. Hãy attach dataset kết quả lần chạy trước."
        )
    dst = REPO / f"outputs_seed_{seed}"
    shutil.copytree(src, dst, dirs_exist_ok=True)
    restored[seed] = dst
    print(f"Seed {seed}: khôi phục từ {src} -> {dst}")
'''

PATCH_PARAMS_CELL = '''# Cập nhật params.yaml theo notebook (đường dẫn, backbone, concept, batch/worker chống OOM)
import yaml

with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)
params["data_dir"] = DATA_DIR
params["num_classes"] = NUM_CLASSES
params["num_epochs"] = NUM_EPOCHS
params["patience"] = PATIENCE
params["batch_size"] = BATCH_SIZE
params["num_workers"] = NUM_WORKERS
params["baseline_comparison"]["selected_architecture"] = BACKBONE
params.setdefault("cbm", {})
params["cbm"]["n_concepts"] = N_CONCEPTS
params["cbm"]["concept_weight"] = CONCEPT_WEIGHT
params["cbm"]["concept_annotations_path"] = CONCEPT_ANNOTATIONS_PATH
with open("params.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(params, f, sort_keys=False, allow_unicode=True)
print("Đã cập nhật params.yaml")
'''

RUN_CELL = '''# Stage 4C + Stage 5 (penalty, head, both) cho 2 seed song song trên 2 GPU
import subprocess, sys
%cd /kaggle/working/neuro_symbolic_mlops_l2_app
subprocess.run(
    [sys.executable, "-m", "pipelines.run_concept_rules_dual_gpu",
     "--config", "params.yaml", "--seeds", str(SEED_A), str(SEED_B), "--modes", *MODES],
    check=True,
)
'''

REPORT_CELL = '''# Bảng so sánh: mô hình cũ (đã train) và mô hình mới (luật từ image_concept)
import pandas as pd
from IPython.display import display

for seed in (SEED_A, SEED_B):
    csv = REPO / f"outputs_seed_{seed}" / "concept_rules_comparison.csv"
    print(f"--- Seed {seed} ---")
    display(pd.read_csv(csv))
'''

PACK_CELL = '''# Đóng gói luật mới và bảng so sánh để tải về
%cd /kaggle/working/neuro_symbolic_mlops_l2_app
!zip -r -q /kaggle/working/concept_rules_outputs.zip outputs_seed_*/04c_concept_rules outputs_seed_*/05c_concept_* outputs_seed_*/concept_rules_comparison.csv outputs_seed_*/execution_concept_rules_*.log
!ls -lh /kaggle/working/concept_rules_outputs.zip
'''


def _cell(kind: str, source: str, tags: list[str] | None = None) -> dict:
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    meta = {"tags": tags} if tags else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [], "source": lines}


def build_notebook(
    template: Path,
    backbone: str,
    seed_a: int,
    seed_b: int,
    data_dir: str,
    concept_path: str,
    num_classes: int,
    n_concepts: int,
    concept_weight: float,
    batch_size: int,
    num_workers: int,
    num_epochs: int,
    patience: int,
    git_repo: str,
    git_branch: str,
    modes: list[str],
) -> dict:
    if not template.exists():
        raise FileNotFoundError(f"Không tìm thấy template: {template}")
    base = json.loads(template.read_text(encoding="utf-8"))

    title = (
        f"# Chạy tiếp: luật GFlowNet từ image_concept — {backbone} — seeds {seed_a}/{seed_b}\n\n"
        "Khôi phục CBM từ lần chạy trước (không train lại Baseline/CBM/RF/GFlowNet cũ), "
        "sau đó chạy **Stage 4C** (GFlowNet xây luật `concept ∧ ¬concept ⇒ class`) và "
        "**Stage 5** ở các mode `penalty`, `head`, `both` trên 2 GPU song song.\n\n"
        "Kết quả: `outputs_seed_*/04c_concept_rules/`, `05c_concept_*/`, "
        "`concept_rules_comparison.csv` và `concept_rules_outputs.zip`."
    )
    cells = [
        _cell("markdown", title),
        _cell("code", PARAMETERS_CELL.format(
            data_dir=data_dir, concept_path=concept_path, num_classes=num_classes,
            n_concepts=n_concepts, backbone=backbone, seed_a=seed_a, seed_b=seed_b,
            concept_weight=concept_weight, batch_size=batch_size, num_workers=num_workers,
            num_epochs=num_epochs, patience=patience, modes=modes,
            git_repo=git_repo, git_branch=git_branch,
        ), tags=["parameters"]),
        _cell("code", CHECK_DATA_CELL),
        _cell("code", CLONE_CELL),
        _cell("code", INSTALL_CELL),
        _cell("code", RESTORE_CELL),
        _cell("code", PATCH_PARAMS_CELL),
        _cell("code", RUN_CELL),
        _cell("code", REPORT_CELL),
        _cell("code", PACK_CELL),
    ]
    base["cells"] = cells
    return base


def generate(args: argparse.Namespace, backbone: str, seed_a: int, seed_b: int) -> Path:
    nb = build_notebook(
        template=Path(args.template),
        backbone=backbone,
        seed_a=seed_a,
        seed_b=seed_b,
        data_dir=args.data_dir,
        concept_path=args.concept_path,
        num_classes=args.num_classes,
        n_concepts=args.n_concepts,
        concept_weight=args.concept_weight,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.epochs,
        patience=args.patience,
        git_repo=args.git_repo,
        git_branch=args.git_branch,
        modes=args.modes,
    )
    out = Path(args.output_dir) / f"concept_rules_{backbone}_seeds_{seed_a}_{seed_b}.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã sinh notebook: {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template", default=str(TEMPLATE))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--backbone", default="mobilenetv3_small")
    p.add_argument("--seeds", nargs=2, type=int, default=[42, 44])
    p.add_argument("--all", action="store_true", help="sinh cho mọi backbone x cặp seed như generate_all.py")
    p.add_argument("--data-dir", default=DEFAULT_DATASET)
    p.add_argument("--concept-path", default=DEFAULT_CONCEPT_PATH)
    p.add_argument("--num-classes", type=int, default=9)
    p.add_argument("--n-concepts", type=int, default=54)
    p.add_argument("--concept-weight", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--git-repo", default=DEFAULT_GIT_REPO)
    p.add_argument("--git-branch", default=DEFAULT_GIT_BRANCH)
    p.add_argument("--modes", nargs="+", default=["penalty", "head", "both"], choices=["penalty", "head", "both"])
    args = p.parse_args()

    if args.all:
        for bb in BACKBONES:
            for a, b in SEED_PAIRS:
                args.backbone = bb
                generate(args, bb, a, b)
    else:
        generate(args, args.backbone, args.seeds[0], args.seeds[1])


if __name__ == "__main__":
    main()
