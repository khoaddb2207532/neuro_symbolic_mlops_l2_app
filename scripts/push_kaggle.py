"""Đẩy notebook lên các account Kaggle.

  python scripts/push_kaggle.py --dry-run            # xem bảng phân công, chưa đẩy
  python scripts/push_kaggle.py                      # đẩy tất cả theo phân công vòng tròn
  python scripts/push_kaggle.py resnet50             # chỉ đẩy notebook có tên chứa chuỗi này
  python scripts/push_kaggle.py --accounts acc1      # chỉ đẩy phần của acc1 (để thử trước)
  python scripts/push_kaggle.py --notebook-dir generated_concept_rules_notebooks \
      --prev-kernel --extra-dataset owner/prev-run-outputs
      # notebook chạy tiếp: đẩy bởi chủ sở hữu notebook cũ và gắn kernel_sources của notebook cũ

--prev-kernel: mỗi notebook concept_rules_<backbone>_seeds_<a>_<b> được đẩy bởi account đã sở
hữu cbm_<backbone>_seeds_<a>_<b> (lấy từ --prev-notebook-dir theo phép chia vòng tròn của lần đẩy
trước), và metadata có "kernel_sources": ["<user>/<slug-cũ>"]. Lý do: kernel riêng tư chỉ đọc được
bởi chính chủ sở hữu, nên không thể gắn kernel của acc1 vào notebook được đẩy bởi acc2.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_assignment import (  # noqa: E402
    build_previous_index,
    owner_assignment,
    round_robin_assignment,
    slug_of,
)

DEFAULT_NB_DIR = "generated_cbm_notebooks"
DATASET = "dangduykhoab2207532/mykhmer"

p = argparse.ArgumentParser()
p.add_argument("names", nargs="*", help="lọc notebook theo chuỗi con của tên file")
p.add_argument("--accounts", nargs="*", help="chỉ đẩy phần của các username này")
p.add_argument("--dry-run", action="store_true")
p.add_argument("--notebook-dir", default=DEFAULT_NB_DIR, help="thư mục chứa notebook cần đẩy")
p.add_argument("--extra-dataset", action="append", default=[],
               help="dataset Kaggle bổ sung (owner/slug), lặp lại được")
p.add_argument("--prev-kernel", action="store_true",
               help="gắn kernel_sources của notebook cũ và đẩy bởi chủ sở hữu của nó")
p.add_argument("--prev-notebook-dir", default=DEFAULT_NB_DIR,
               help="thư mục notebook CBM cũ, dùng để suy ra kernel nguồn (chỉ với --prev-kernel)")
p.add_argument("--wait", type=float, default=60.0,
               help="số giây chờ sau mỗi lần push notebook (mặc định 60; 0 để tắt)")
args = p.parse_args()

accounts = json.loads(Path(".kaggle/accounts.json").read_text(encoding="utf-8"))
all_nbs = sorted(Path(args.notebook_dir).glob("*.ipynb"))

if args.prev_kernel:
    prev_nbs = sorted(Path(args.prev_notebook_dir).glob("*.ipynb"))
    prev_index = build_previous_index(prev_nbs, accounts)
    assignment = owner_assignment(all_nbs, prev_index, accounts)  # (account, notebook, kernel)
else:
    assignment = [(acc, nb, None) for acc, nb in round_robin_assignment(all_nbs, accounts)]

jobs = [(a, nb, k) for a, nb, k in assignment
        if (not args.names or any(s in nb.stem for s in args.names))
        and (not args.accounts or a["username"] in args.accounts)]

print(f"{len(all_nbs)} notebook / {len(accounts)} account. Sẽ đẩy {len(jobs)}:")
for a, nb, kernel in jobs:
    src = f"  <- kernel {kernel}" if kernel else ""
    print(f"  {a['username']:<15} <- {nb.name}{src}")
if args.dry_run:
    raise SystemExit(0)

failed = []
for acc, nb, kernel in jobs:
    slug = slug_of(nb.stem)
    env = {k: v for k, v in os.environ.items()
           if k not in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN")}
    env["KAGGLE_API_TOKEN"] = acc["token"]
    env["KAGGLE_USERNAME"] = acc["username"]
    print(f"\n[{acc['username']}] {slug}")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copy(nb, work / nb.name)
        meta = {
            "id": f"{acc['username']}/{slug}",
            "title": slug,
            "code_file": nb.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,   # notebook cần git clone repo
            "dataset_sources": [DATASET, *args.extra_dataset],
            "competition_sources": [],
        }
        if kernel:
            meta["kernel_sources"] = [kernel]
        (work / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(work)], env=env)
        if r.returncode != 0:
            failed.append(f"{acc['username']}/{slug}")
    if args.wait > 0 and (acc, nb, kernel) != jobs[-1]:
        print(f"  chờ {args.wait:g} giây trước notebook tiếp theo...")
        time.sleep(args.wait)

if failed:
    print("\nLỖI:", *failed, sep="\n  ")
    raise SystemExit(1)
print("\nXong.")
