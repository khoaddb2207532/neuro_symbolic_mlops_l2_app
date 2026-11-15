"""Chia đều các notebook trong generated_cbm_notebooks/ lên các account Kaggle.

  python scripts/push_kaggle.py --dry-run            # xem bảng phân công, chưa đẩy
  python scripts/push_kaggle.py                      # đẩy tất cả theo phân công
  python scripts/push_kaggle.py resnet50             # chỉ đẩy notebook có tên chứa chuỗi này
  python scripts/push_kaggle.py --accounts acc1      # chỉ đẩy phần của acc1 (để thử trước)
"""
import argparse, json, os, shutil, subprocess, tempfile
from pathlib import Path

NB_DIR = Path("generated_cbm_notebooks")
DATASET = "dangduykhoab2207532/mykhmer"

p = argparse.ArgumentParser()
p.add_argument("names", nargs="*", help="lọc notebook theo chuỗi con của tên file")
p.add_argument("--accounts", nargs="*", help="chỉ đẩy phần của các username này")
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()

accounts = json.loads(Path(".kaggle/accounts.json").read_text(encoding="utf-8"))
all_nbs = sorted(NB_DIR.glob("*.ipynb"))

# Phân công dựa trên TOÀN BỘ danh sách để không đổi khi bạn lọc
base, extra = divmod(len(all_nbs), len(accounts))
assignment, start = [], 0
for i, acc in enumerate(accounts):
    n = base + (1 if i < extra else 0)
    for nb in all_nbs[start:start + n]:
        assignment.append((acc, nb))
    start += n

# Lọc theo yêu cầu
jobs = [(a, nb) for a, nb in assignment
        if (not args.names or any(s in nb.stem for s in args.names))
        and (not args.accounts or a["username"] in args.accounts)]

print(f"{len(all_nbs)} notebook / {len(accounts)} account. Sẽ đẩy {len(jobs)}:")
for a, nb in jobs:
    print(f"  {a['username']:<15} <- {nb.name}")
if args.dry_run:
    raise SystemExit(0)

failed = []
for acc, nb in jobs:
    slug = nb.stem.lower().replace("_", "-")
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
            "dataset_sources": [DATASET],
            "competition_sources": [],
        }
        (work / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(work)], env=env)
        if r.returncode != 0:
            failed.append(f"{acc['username']}/{slug}")

if failed:
    print("\nLỖI:", *failed, sep="\n  ")
    raise SystemExit(1)
print("\nXong.")